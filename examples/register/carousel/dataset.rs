//! The register-carousel dataset: a three-register file with **one input port**
//! and a **turn** instruction that permutes the registers under it.
//!
//! ```text
//!   turn = rotate            + R + R - R ? R ? R ?
//!   register A (read port)   +   .   +   -   +   .   -
//!   register B               .   +   .   +   .   -   .
//!   register C               .   .   .   +   .   +   .
//!   target                   .   .   .   .   .   Neg .   Pos .   Neg
//! ```
//!
//! `+` / `-` overwrite register A with a bit, `R` turns the carousel, and `?`
//! reads register A back. Only `?` positions are scored, and only once A holds
//! something (see [`IGNORE`]).
//!
//! ## Why this needs more than one Householder
//!
//! A turn is a **permutation of the state's rows**, and the delta rule applies
//! exactly one generalised Householder `α(I − β k kᵀ)` per micro-step. With
//! `‖k‖ = 1` that matrix's eigenvalues are `{α, α, α(1−β)}` — all **real**.
//!
//! - [`Turn::Swap`] is a transposition: `I − 2wwᵀ` with `w = (e_A − e_B)/√2`
//!   *is* the swap, exactly. It needs `β = 2`, which is what `allow_neg_eigval`
//!   admits; with `β ≤ 1` the spectrum is `[0, 1]` and nothing can oscillate at
//!   all.
//! - [`Turn::Rotate`] is a 3-cycle. Its eigenvalues are `1, e^{±2πi/3}` —
//!   **complex**, so no single real Householder is that matrix, and no orbit of
//!   one has period 3. Two of them do: the 3-cycle is `swap(A,C)∘swap(A,B)`,
//!   which is exactly what `n_householder = 2` provides per token.
//!
//! [`Family::Cycle`] is where that bites: it loads the file and then reads the
//! port once per turn, so the answer marches around the carousel and a model
//! that cannot hold the period is wrong on a third of the queries (`Rotate`) or
//! half of them (`Swap`). See `tests.rs` for the sweep over every single
//! Householder.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Input symbol: overwrite register A with `−1`.
pub const PUSH_NEG: usize = 0;
/// Input symbol: overwrite register A with `+1`.
pub const PUSH_POS: usize = 1;
/// Input symbol: turn the carousel.
pub const TURN: usize = 2;
/// Input symbol: read register A.
pub const QUERY: usize = 3;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = 4;

/// Number of registers in the file.
pub const NUM_REGISTERS: usize = 3;

/// Target class: register A holds `−1`.
pub const NEG: i64 = 0;
/// Target class: register A holds `+1`.
pub const POS: i64 = 1;
/// Number of output classes.
pub const NUM_CLASSES: usize = 2;

/// Placeholder target for a position with **nothing to report** — every
/// non-query position, and every query issued while register A is still empty.
pub const IGNORE: i64 = -1;

/// Length of every generated sequence.
pub const SEQ_LENGTH: usize = 32;
/// Number of training sequences.
pub const NUM_TRAIN: usize = 4096;
/// Number of evaluation sequences (per family).
pub const NUM_EVAL: usize = 512;

/// Dataset RNG seed for the training split.
pub const TRAIN_SEED: u64 = 0xCA6015;
/// Dataset RNG seed for the evaluation splits (distinct from training).
pub const EVAL_SEED: u64 = 0xBEEF;

/// What the `R` instruction does to the register file.
///
/// The two differ only in the *order* of the permutation, and that is the whole
/// contrast: see the module header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Turn {
    /// Swap registers A and B; C is never reached. One reflection — reachable
    /// with a single Householder, provided `β` may reach 2.
    Swap,
    /// Cycle `A ← C ← B ← A`. A 3-cycle: two reflections, never one.
    Rotate,
}

impl Turn {
    /// How many registers the turn can reach from the input port.
    pub fn reachable(self) -> usize {
        match self {
            Self::Swap => 2,
            Self::Rotate => 3,
        }
    }

    /// Apply the turn to a register file.
    pub fn apply<T: Copy>(self, file: [T; NUM_REGISTERS]) -> [T; NUM_REGISTERS] {
        match self {
            Self::Swap => [file[1], file[0], file[2]],
            Self::Rotate => [file[2], file[0], file[1]],
        }
    }

    /// Parse the `--turn` CLI value.
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some("rotate") | Some("cycle") | None => Self::Rotate,
            Some("swap") => Self::Swap,
            Some(other) => panic!("--turn must be 'rotate' or 'swap', got {other:?}"),
        }
    }
}

/// The per-position targets implied by a symbol sequence under `turn`.
pub fn labels(symbols: &[usize], turn: Turn) -> Vec<i64> {
    let mut file = [None::<i64>; NUM_REGISTERS];
    symbols
        .iter()
        .map(|&s| match s {
            PUSH_NEG | PUSH_POS => {
                file[0] = Some(if s == PUSH_POS { 1 } else { -1 });
                IGNORE
            }
            TURN => {
                file = turn.apply(file);
                IGNORE
            }
            QUERY => match file[0] {
                None => IGNORE,
                Some(bit) => {
                    if bit > 0 {
                        POS
                    } else {
                        NEG
                    }
                }
            },
            _ => panic!("symbol out of alphabet: {s}"),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: pushes, turns and queries mixed.
    Random,
    /// Load the whole file, then read the port once per turn — so the answer
    /// marches around the carousel and the query stream carries the file's
    /// period. This is the family a single Householder cannot hold.
    Cycle,
    /// The training mixture: half of each.
    Mixed,
}

/// SplitMix64 — a small deterministic RNG so splits reproduce exactly.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn push(&mut self) -> usize {
        self.below(2)
    }
}

fn gen_random(rng: &mut Lcg, len: usize) -> Vec<usize> {
    (0..len)
        .map(|_| match rng.below(20) {
            0..=6 => TURN,
            7..=12 => QUERY,
            _ => rng.push(),
        })
        .collect()
}

fn gen_cycle(rng: &mut Lcg, len: usize, turn: Turn) -> Vec<usize> {
    let slots = turn.reachable();
    // Load every reachable register with a bit, one push per turn. Not all the
    // same: a constant file would be answered by a constant output.
    let mut bits: Vec<usize> = (0..slots).map(|_| rng.push()).collect();
    if bits.iter().all(|b| *b == bits[0]) {
        bits[rng.below(slots)] ^= 1;
    }
    let mut out = Vec::with_capacity(len);
    for (i, bit) in bits.iter().enumerate() {
        if i > 0 {
            out.push(TURN);
        }
        out.push(*bit);
    }
    // ... then read the port once per turn, for the rest of the sequence.
    while out.len() < len {
        out.push(QUERY);
        out.push(TURN);
    }
    out.truncate(len);
    out
}

/// Generate one sequence of the given family.
pub fn generate(family: Family, turn: Turn, rng_state: &mut u64, len: usize) -> Vec<usize> {
    let mut rng = Lcg(*rng_state);
    let out = match family {
        Family::Random => gen_random(&mut rng, len),
        Family::Cycle => gen_cycle(&mut rng, len, turn),
        Family::Mixed => {
            if rng.below(2) == 0 {
                gen_cycle(&mut rng, len, turn)
            } else {
                gen_random(&mut rng, len)
            }
        }
    };
    *rng_state = rng.0;
    out
}

// ---------------------------------------------------------------------------
// Dataset / batcher
// ---------------------------------------------------------------------------

/// One generated sequence and its per-position target class.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarouselItem {
    /// Input symbols.
    pub symbols: Vec<usize>,
    /// Per-position target class ([`NEG`] / [`POS`] / [`IGNORE`]).
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`CarouselItem`]s.
pub struct CarouselDataset {
    dataset: InMemDataset<CarouselItem>,
}

impl CarouselDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(
        num_sequences: usize,
        seq_length: usize,
        family: Family,
        turn: Turn,
        seed: u64,
    ) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, turn, &mut state, seq_length);
                let targets = labels(&symbols, turn);
                CarouselItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<CarouselItem> for CarouselDataset {
    fn get(&self, index: usize) -> Result<CarouselItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`CarouselItem`]s into a [`CarouselBatch`], one-hotting the symbols.
#[derive(Clone, Debug, Default)]
pub struct CarouselBatcher {}

/// A batch of one-hot symbol sequences and their per-position target classes.
#[derive(Clone, Debug)]
pub struct CarouselBatch {
    /// One-hot input symbol at each position, `[batch, seq, NUM_SYMBOLS]`.
    pub inputs: Tensor<3>,
    /// Per-position target class, `[batch, seq]`.
    pub targets: Tensor<2, Int>,
    /// Flat indices (row-major over `batch × seq`) of the scored positions.
    pub scored: Tensor<1, Int>,
}

/// One-hot encode a symbol sequence into `[seq, NUM_SYMBOLS]`.
pub fn one_hot(symbols: &[usize], device: &Device) -> Tensor<2> {
    let mut buf = vec![0.0f32; symbols.len() * NUM_SYMBOLS];
    for (t, &s) in symbols.iter().enumerate() {
        buf[t * NUM_SYMBOLS + s] = 1.0;
    }
    Tensor::<1>::from_floats(buf.as_slice(), device).reshape([symbols.len(), NUM_SYMBOLS])
}

impl Batcher<CarouselItem, CarouselBatch> for CarouselBatcher {
    fn batch(&self, items: Vec<CarouselItem>, device: &Device) -> CarouselBatch {
        let inputs: Vec<Tensor<2>> = items
            .iter()
            .map(|item| one_hot(&item.symbols, device))
            .collect();
        let targets: Vec<Tensor<1, Int>> = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        let seq = items[0].symbols.len();
        let scored: Vec<i32> = items
            .iter()
            .enumerate()
            .flat_map(|(b, item)| {
                item.targets
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c != IGNORE)
                    .map(move |(t, _)| (b * seq + t) as i32)
            })
            .collect();
        CarouselBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
            scored: Tensor::<1, Int>::from_ints(scored.as_slice(), device),
        }
    }
}

/// Render a symbol sequence as `+ R - ? …`.
pub fn render_symbols(symbols: &[usize]) -> String {
    symbols
        .iter()
        .map(|&s| match s {
            PUSH_NEG => '-',
            PUSH_POS => '+',
            TURN => 'R',
            _ => '?',
        })
        .collect()
}

/// Render per-position classes as `n`/`p`/`.`, aligned with [`render_symbols`].
pub fn render_classes(classes: &[i64]) -> String {
    classes
        .iter()
        .map(|&c| match c {
            NEG => 'n',
            POS => 'p',
            _ => '.',
        })
        .collect()
}
