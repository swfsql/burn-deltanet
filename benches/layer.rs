//! # Single-block benchmarks (`cargo bench`)
//!
//! One delta-rule block per case — no `Layer`/`Layers`/network wrapper —
//! measured in the three modes every block exposes:
//!
//! | Group | What it runs |
//! |-------|--------------|
//! | `forward` | `forward` on a plain device (chunkwise prefill / inference) |
//! | `train`   | `forward` + `loss.backward()` on an autodiff device |
//! | `step`    | one recurrent `step` from the previous step's cache (decode) |
//!
//! Cases cover the axes that have their own code path: the four families, the
//! two `TriSolve` variants of the WY inverse, `Recurrent` against `Chunk`,
//! DeltaProduct's `n_householder`, and — for GDN-2 — the per-channel decay,
//! whose intra-chunk score matrices go through the block-decomposed path in
//! `delta::decay` instead of a plain `[chunk_len, chunk_len]` mask.
//!
//! ## Running
//!
//! ```bash
//! cargo bench                                   # default features (flex)
//!
//! # a GPU backend as it is actually deployed:
//! BURN_DEVICE=cuda cargo bench --features "backend-cuda,fusion,dev-autotune"
//!
//! # regression tracking (criterion stores baselines under target/criterion):
//! cargo bench -- --save-baseline flex
//! cargo bench -- --baseline flex                # report % change vs. that run
//!
//! cargo bench -- forward/gated                  # one case
//! ```
//!
//! `BURN_DEVICE` picks the backend when several are compiled in, so one build
//! benches both flex and CUDA; only kernel fusion, being compile-time, needs a
//! build of its own.
//!
//! Every case runs [`warmup_iters`] untimed iterations first, so kernel
//! compilation and autotuning are finished before criterion measures anything.
//! Each measured *batch* then submits all its iterations and drains the device
//! once at the end ([`timed`]) — an async backend is measured at steady state,
//! not one submit-drain round trip at a time.
//!
//! The block, its input and that warm-up are built inside the closure criterion
//! only calls for cases that pass its filter, so `-- deltanet` really does touch
//! nothing else.
//!
//! ## Sizing
//!
//! Defaults are small enough to finish on the CPU backends and still large
//! enough to be GEMM-bound on a GPU. Override per run with the environment:
//! `BENCH_BATCH`, `BENCH_SEQ`, `BENCH_D_MODEL`, `BENCH_HEAD_K_DIM`,
//! `BENCH_NHEADS`, `BENCH_CHUNK_LEN`, plus `BENCH_SAMPLES` / `BENCH_TIME_MS`
//! for criterion's sampling and `BENCH_WARMUP_ITERS` / `BENCH_SYNC_EVERY` for
//! the warm-up and drain policy.
//!
//! ```bash
//! BENCH_SEQ=2048 BENCH_D_MODEL=1024 cargo bench --features backend-cuda -- forward
//! ```

use burn::prelude::*;
use burn::tensor::Distribution;
use burn_deltanet::prelude::*;
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The problem size every case is measured at (shared, so the numbers are
/// comparable across families).
#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: usize,
    sequence: usize,
    d_model: usize,
    nheads: usize,
    head_k_dim: usize,
    chunk_len: usize,
}

impl Shape {
    fn from_env() -> Self {
        Self {
            batch: env_usize("BENCH_BATCH", 2),
            sequence: env_usize("BENCH_SEQ", 256),
            d_model: env_usize("BENCH_D_MODEL", 256),
            nheads: env_usize("BENCH_NHEADS", 4),
            head_k_dim: env_usize("BENCH_HEAD_K_DIM", 64),
            chunk_len: env_usize("BENCH_CHUNK_LEN", 64),
        }
    }

    /// Tokens per `forward` / `train` iteration (criterion reports elem/s).
    fn tokens(&self) -> u64 {
        (self.batch * self.sequence) as u64
    }

    /// Print the effective configuration once per process, so a bench log is
    /// self-describing.
    fn announce(&self, device: &Device) {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            // `Backend::name` nests the wrappers that are *compiled in*, e.g.
            // `dispatch<fusion<cubecl<cuda>>>` vs `dispatch<cubecl<cuda>>`, so
            // the log proves which flavour ran instead of trusting the feature
            // flags.
            let backend =
                <burn::backend::Dispatch as burn::backend::Backend>::name(device.as_dispatch());
            eprintln!(
                "bench-config: batch={} sequence={} d_model={} nheads={} head_k_dim={} \
                 chunk_len={} warmup_iters={} backend={backend}",
                self.batch,
                self.sequence,
                self.d_model,
                self.nheads,
                self.head_k_dim,
                self.chunk_len,
                warmup_iters(),
            );
        });
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{key}: expected an integer, got {v:?}"))
        })
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Devices and timing
// ---------------------------------------------------------------------------

/// Block until every queued operation has actually run.
///
/// The GPU backends are asynchronous: without this a measured iteration would
/// only time the op *submission*, and the real work would land in whichever
/// iteration happens to synchronise next.
fn sync(device: &Device) {
    device.sync().expect("device sync failed");
}

/// Time `iters` iterations of `work`, draining the device **once at the end**.
///
/// Syncing inside every iteration would measure submit-then-drain latency and
/// serialise the queue — the CPU would wait for each kernel before submitting
/// the next, which is not how a training loop feeds a GPU. The drain stays
/// *inside* the timed region, so no work escapes the measurement; it just
/// amortises over the batch. `BENCH_SYNC_EVERY=N` drains every `N` iterations
/// instead (`0`, the default, drains only at the end).
fn timed<T>(device: &Device, iters: u64, mut work: impl FnMut() -> T) -> Duration {
    let sync_every = env_usize("BENCH_SYNC_EVERY", 0) as u64;
    let start = Instant::now();
    for i in 0..iters {
        black_box(work());
        if sync_every != 0 && (i + 1) % sync_every == 0 {
            sync(device);
        }
    }
    sync(device);
    start.elapsed()
}

/// How many untimed iterations to run before criterion starts measuring.
///
/// A cubecl backend compiles a kernel on its first execution for a given shape,
/// and with `dev-autotune` it also *tunes* it then — one-off costs that must not
/// land in a measured sample.
fn warmup_iters() -> usize {
    env_usize("BENCH_WARMUP_ITERS", 2)
}

fn configure(group: &mut BenchmarkGroup<'_, WallTime>, tokens: u64) {
    let time_ms = env_usize("BENCH_TIME_MS", 5000) as u64;
    group.throughput(Throughput::Elements(tokens));
    group.sample_size(env_usize("BENCH_SAMPLES", 10));
    group.warm_up_time(Duration::from_millis(time_ms / 5));
    group.measurement_time(Duration::from_millis(time_ms));
}

fn input_3d(shape: Shape, device: &Device) -> Tensor<3> {
    Tensor::random(
        [shape.batch, shape.sequence, shape.d_model],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn input_2d(shape: Shape, device: &Device) -> Tensor<2> {
    Tensor::random(
        [shape.batch, shape.d_model],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

// ---------------------------------------------------------------------------
// Case configurations
// ---------------------------------------------------------------------------

fn deltanet_config(shape: Shape) -> DeltaNetConfig {
    DeltaNetConfig::new(shape.d_model).with_nheads(shape.nheads)
}

fn gated_config(shape: Shape) -> GatedDeltaNetConfig {
    GatedDeltaNetConfig::new(shape.d_model)
        .with_nheads(shape.nheads)
        .with_head_k_dim(shape.head_k_dim)
        .with_expand_v(1.0)
}

fn gdn2_config(shape: Shape) -> GatedDeltaNet2Config {
    GatedDeltaNet2Config::new(shape.d_model)
        .with_nheads(shape.nheads)
        .with_head_k_dim(shape.head_k_dim)
        .with_expand_v(1.0)
}

fn product_config(shape: Shape, n_householder: usize) -> DeltaProductConfig {
    DeltaProductConfig::new(shape.d_model)
        .with_nheads(shape.nheads)
        .with_head_k_dim(shape.head_k_dim)
        .with_expand_v(1.0)
        .with_n_householder(n_householder)
}

/// Every `(name, path)` the chunk algorithm is measured under.
///
/// `Recurrent` is included deliberately: it is the same function, and the gap
/// between the two is the entire justification for the WY machinery.
fn paths(shape: Shape) -> Vec<(&'static str, DeltaPath)> {
    vec![
        (
            "chunk-doubling",
            DeltaPath::Chunk {
                chunk_len: Some(shape.chunk_len),
                solve: TriSolve::Doubling,
            },
        ),
        (
            "chunk-neumann",
            DeltaPath::Chunk {
                chunk_len: Some(shape.chunk_len),
                solve: TriSolve::Neumann,
            },
        ),
        ("recurrent", DeltaPath::Recurrent),
    ]
}

// ---------------------------------------------------------------------------
// The three groups
// ---------------------------------------------------------------------------

/// Register one `forward` case. The block and its input are built inside the
/// closure, so a filtered-out case allocates nothing.
macro_rules! forward_case {
    ($group:expr, $name:expr, $shape:expr, $device:expr, $config:expr, $path:expr) => {{
        let (shape, device, path) = ($shape, $device.clone(), $path);
        $group.bench_function($name, |b| {
            let block = $config.init(&device);
            let x = input_3d(shape, &device);
            for _ in 0..warmup_iters() {
                black_box(block.forward(x.clone(), None, path));
            }
            sync(&device);
            b.iter_custom(|iters| {
                timed(&device, iters, || block.forward(x.clone(), None, path))
            });
        });
    }};
}

/// Register one `train` case: the same forward, plus a backward through a
/// scalar loss.
macro_rules! train_case {
    ($group:expr, $name:expr, $shape:expr, $device:expr, $config:expr, $path:expr) => {{
        let (shape, device, path) = ($shape, $device.clone(), $path);
        $group.bench_function($name, |b| {
            let block = $config.init(&device);
            let x = input_3d(shape, &device);
            let run = || {
                let (y, _cache) = block.forward(x.clone(), None, path);
                y.powi_scalar(2).mean().backward()
            };
            for _ in 0..warmup_iters() {
                black_box(run());
            }
            sync(&device);
            b.iter_custom(|iters| timed(&device, iters, run));
        });
    }};
}

/// Register one `step` case: one recurrent token from the previous step's
/// cache, which is what decoding actually costs.
macro_rules! step_case {
    ($group:expr, $name:expr, $shape:expr, $device:expr, $config:expr) => {{
        let (shape, device) = ($shape, $device.clone());
        $group.bench_function($name, |b| {
            let block = $config.init(&device);
            let x = input_2d(shape, &device);
            let mut cache = None;
            for _ in 0..warmup_iters() {
                let (_y, next) = block.step(x.clone(), cache.take());
                cache = Some(next);
            }
            sync(&device);
            b.iter_custom(|iters| {
                timed(&device, iters, || {
                    let (y, next) = block.step(x.clone(), cache.take());
                    cache = Some(next);
                    y
                })
            });
        });
    }};
}

fn bench_forward(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = Device::default();
    shape.announce(&device);

    let mut group = c.benchmark_group("forward");
    configure(&mut group, shape.tokens());

    for (path_name, path) in paths(shape) {
        forward_case!(
            group,
            format!("deltanet/{path_name}"),
            shape,
            device,
            deltanet_config(shape),
            path
        );
        forward_case!(
            group,
            format!("gated/{path_name}"),
            shape,
            device,
            gated_config(shape),
            path
        );
        forward_case!(
            group,
            format!("gdn2/{path_name}"),
            shape,
            device,
            gdn2_config(shape),
            path
        );
    }

    // `n_householder` multiplies the core's sequence, so its cost shows up here
    // and nowhere else — the state stays the same size.
    let path = DeltaPath::chunk_len(shape.chunk_len);
    for u in [1, 2, 3] {
        forward_case!(
            group,
            format!("product-{u}/chunk-doubling"),
            shape,
            device,
            product_config(shape, u),
            path
        );
    }

    group.finish();
}

fn bench_train(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = Device::default().autodiff();
    shape.announce(&device);

    let mut group = c.benchmark_group("train");
    configure(&mut group, shape.tokens());

    let path = DeltaPath::chunk_len(shape.chunk_len);
    train_case!(group, "deltanet", shape, device, deltanet_config(shape), path);
    train_case!(group, "gated", shape, device, gated_config(shape), path);
    train_case!(group, "gdn2", shape, device, gdn2_config(shape), path);
    train_case!(
        group,
        "product-2",
        shape,
        device,
        product_config(shape, 2),
        path
    );

    // The Neumann solve is the reference, and its `chunk_len − 1` matmuls are
    // also `chunk_len − 1` retained activations — the memory cost the doubling
    // factorisation exists to avoid.
    train_case!(
        group,
        "gated/chunk-neumann",
        shape,
        device,
        gated_config(shape),
        DeltaPath::Chunk {
            chunk_len: Some(shape.chunk_len),
            solve: TriSolve::Neumann,
        }
    );

    group.finish();
}

fn bench_step(c: &mut Criterion) {
    let shape = Shape::from_env();
    let device = Device::default();
    shape.announce(&device);

    let mut group = c.benchmark_group("step");
    // One token per iteration, whatever the sequence length is set to.
    configure(&mut group, shape.batch as u64);

    step_case!(group, "deltanet", shape, device, deltanet_config(shape));
    step_case!(group, "gated", shape, device, gated_config(shape));
    step_case!(group, "gdn2", shape, device, gdn2_config(shape));
    for u in [1, 2, 3] {
        step_case!(
            group,
            format!("product-{u}"),
            shape,
            device,
            product_config(shape, u)
        );
    }

    group.finish();
}

criterion_group!(benches, bench_forward, bench_train, bench_step);
criterion_main!(benches);
