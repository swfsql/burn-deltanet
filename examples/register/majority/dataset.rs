//! The register-majority dataset: a stream of **writes** to a three-register
//! file, and **queries** asking for the majority of what the registers hold
//! *right now*.
//!
//! ```text
//!   symbols   a+  b+  c-  ?   a-  ?   c+  ?   b-  ?
//!   registers a+  a+  a+  a+  a-  a-  a-  a-  a-  a-
//!             .   b+  b+  b+  b+  b+  b+  b+  b-  b-
//!             .   .   c-  c-  c-  c-  c+  c+  c+  c+
//!   target    .   .   .   Pos .   Neg .   Pos .   Neg
//! ```
//!
//! Only `?` positions are scored, and only once every register has been
//! written at least once — before that there is no majority to report (see
//! [`IGNORE`]). Three `±1` bits never tie, so the answer is always one of two
//! classes and no calibration band is being tested.
//!
//! Two properties make this the task a single DeltaNet block is for:
//!
//! - **It needs the state, and a keyed one.** A register may have been written
//!   arbitrarily far back, and the answer is not a function of the last symbol
//!   (`?` carries no information at all). The block runs with no short
//!   convolution, so the recurrent state is the model's only memory.
//! - **It needs the write to *erase*.** Linear attention accumulates:
//!   `S ← αS + k vᵀ` piles every value ever written at a key on top of the
//!   others, and the only way to remove one is a decay that removes everything.
//!   [`Family::Stale`] and [`Family::Fading`] are the two adversarial halves
//!   that pin that down — the first defeats any `α` near 1 (a register's
//!   *history* outvotes its present), the second defeats any `α` away from 1
//!   (the oldest register's vote fades before it is read). See `tests.rs`.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Number of registers in the file.
pub const NUM_REGISTERS: usize = 3;
/// Input symbol: the query, `?`. Everything below it is a write.
pub const QUERY: usize = 2 * NUM_REGISTERS;
/// Input alphabet size: two write symbols per register, plus the query.
pub const NUM_SYMBOLS: usize = QUERY + 1;

/// The write symbol setting register `reg` to `bit` (`false` ⇒ `−`).
pub const fn write_symbol(reg: usize, bit: bool) -> usize {
    2 * reg + bit as usize
}

/// Target class: the three registers hold a negative majority.
pub const NEG: i64 = 0;
/// Target class: the three registers hold a positive majority.
pub const POS: i64 = 1;
/// Number of output classes.
pub const NUM_CLASSES: usize = 2;

/// Placeholder target for a position with **nothing to report** — every write
/// position, and every query issued before all three registers are loaded.
///
/// Those positions are dropped from the loss and from the accuracy (the batcher
/// emits [`RegisterBatch::scored`] for the rest).
pub const IGNORE: i64 = -1;

/// Length of every generated sequence.
pub const SEQ_LENGTH: usize = 32;
/// Number of training sequences.
pub const NUM_TRAIN: usize = 4096;
/// Number of evaluation sequences (per family).
pub const NUM_EVAL: usize = 512;

/// Dataset RNG seed for the training split.
pub const TRAIN_SEED: u64 = 0xC0FFEE;
/// Dataset RNG seed for the evaluation splits (distinct from training).
pub const EVAL_SEED: u64 = 0xBEEF;

/// The per-position targets implied by a symbol sequence.
///
/// A write overwrites its register outright; a query reports the sign of the
/// three registers' sum, or [`IGNORE`] while any of them is still empty.
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    let mut file = [None::<i64>; NUM_REGISTERS];
    symbols
        .iter()
        .map(|&s| {
            if s == QUERY {
                let sum: Option<i64> = file.iter().copied().sum();
                match sum {
                    None => IGNORE,
                    Some(sum) => {
                        debug_assert_ne!(sum, 0, "an odd number of ±1 bits cannot tie");
                        if sum > 0 { POS } else { NEG }
                    }
                }
            } else {
                assert!(s < QUERY, "symbol out of alphabet: {s}");
                file[s / 2] = Some(if s % 2 == 0 { -1 } else { 1 });
                IGNORE
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: a write with probability ~⅗, otherwise a query.
    Random,
    /// Every register written **four times**, three of them with the opposite
    /// bit, so a register's history says the opposite of its present. Defeats
    /// any decay close to 1 (the superseded writes are still in the sum).
    Stale,
    /// The three writes spread far apart, with the most recent one the odd bit
    /// out half the time. Defeats any decay away from 1 (the oldest register's
    /// vote has faded by the time the query arrives).
    Fading,
    /// The training mixture: half [`Self::Random`], a quarter of each
    /// adversarial family.
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
    fn bit(&mut self) -> bool {
        self.below(2) == 1
    }
}

fn gen_random(rng: &mut Lcg, len: usize) -> Vec<usize> {
    (0..len)
        .map(|_| {
            if rng.below(5) < 3 {
                rng.below(QUERY)
            } else {
                QUERY
            }
        })
        .collect()
}

fn gen_stale(rng: &mut Lcg, len: usize) -> Vec<usize> {
    assert!(len >= 4 * NUM_REGISTERS, "Stale needs room to load the file");
    let mut out = Vec::with_capacity(len);
    for reg in 0..NUM_REGISTERS {
        let bit = rng.bit();
        // three writes of the wrong bit, then the right one: any un-erased
        // accumulation of this register's writes has the opposite sign.
        out.extend(std::iter::repeat_n(write_symbol(reg, !bit), 3));
        out.push(write_symbol(reg, bit));
    }
    while out.len() < len {
        out.push(if rng.bit() { QUERY } else { rng.below(QUERY) });
    }
    out.truncate(len);
    out
}

fn gen_fading(rng: &mut Lcg, len: usize) -> Vec<usize> {
    const GAP: usize = 6;
    assert!(
        len >= NUM_REGISTERS * (GAP + 1),
        "Fading needs room to spread the writes"
    );
    // Two registers carry `majority`; the third — written **last** — carries the
    // odd bit out half the time, so reading the freshest write alone is right
    // only by coin flip while the true answer is always `majority`.
    let majority = rng.bit();
    let odd_one_out = if rng.bit() { !majority } else { majority };
    let mut order: Vec<usize> = (0..NUM_REGISTERS).collect();
    for i in (1..NUM_REGISTERS).rev() {
        order.swap(i, rng.below(i + 1));
    }

    let mut out = Vec::with_capacity(len);
    for (i, &reg) in order.iter().enumerate() {
        let bit = if i + 1 < NUM_REGISTERS {
            majority
        } else {
            odd_one_out
        };
        out.push(write_symbol(reg, bit));
        out.extend(std::iter::repeat_n(QUERY, GAP));
    }
    while out.len() < len {
        out.push(QUERY);
    }
    out.truncate(len);
    out
}

/// Generate one sequence of the given family.
pub fn generate(family: Family, rng_state: &mut u64, len: usize) -> Vec<usize> {
    let mut rng = Lcg(*rng_state);
    let out = match family {
        Family::Random => gen_random(&mut rng, len),
        Family::Stale => gen_stale(&mut rng, len),
        Family::Fading => gen_fading(&mut rng, len),
        Family::Mixed => match rng.below(4) {
            0 => gen_stale(&mut rng, len),
            1 => gen_fading(&mut rng, len),
            _ => gen_random(&mut rng, len),
        },
    };
    *rng_state = rng.0;
    out
}

// ---------------------------------------------------------------------------
// Dataset / batcher
// ---------------------------------------------------------------------------

/// One generated sequence and its per-position target class.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterItem {
    /// Input symbols: a write (`< QUERY`) or [`QUERY`].
    pub symbols: Vec<usize>,
    /// Per-position target class ([`NEG`] / [`POS`] / [`IGNORE`]).
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`RegisterItem`]s.
pub struct RegisterDataset {
    dataset: InMemDataset<RegisterItem>,
}

impl RegisterDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(num_sequences: usize, seq_length: usize, family: Family, seed: u64) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, &mut state, seq_length);
                let targets = labels(&symbols);
                RegisterItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<RegisterItem> for RegisterDataset {
    fn get(&self, index: usize) -> Result<RegisterItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`RegisterItem`]s into a [`RegisterBatch`], one-hotting the symbols.
#[derive(Clone, Debug, Default)]
pub struct RegisterBatcher {}

/// A batch of one-hot symbol sequences and their per-position target classes.
#[derive(Clone, Debug)]
pub struct RegisterBatch {
    /// One-hot input symbol at each position, `[batch, seq, NUM_SYMBOLS]`.
    pub inputs: Tensor<3>,
    /// Per-position target class, `[batch, seq]`; [`IGNORE`] where there is
    /// nothing to report.
    pub targets: Tensor<2, Int>,
    /// Flat indices (row-major over `batch × seq`) of the scored positions —
    /// everything but the [`IGNORE`]s.
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

impl Batcher<RegisterItem, RegisterBatch> for RegisterBatcher {
    fn batch(&self, items: Vec<RegisterItem>, device: &Device) -> RegisterBatch {
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
        RegisterBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
            scored: Tensor::<1, Int>::from_ints(scored.as_slice(), device),
        }
    }
}

/// Render a symbol sequence as `a+ b- ?  …`, one two-character cell per symbol.
pub fn render_symbols(symbols: &[usize]) -> String {
    symbols
        .iter()
        .map(|&s| {
            if s == QUERY {
                "? ".to_string()
            } else {
                format!("{}{}", (b'a' + (s / 2) as u8) as char, if s % 2 == 0 { '-' } else { '+' })
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render per-position classes as `n`/`p`/`.`, aligned with [`render_symbols`].
pub fn render_classes(classes: &[i64]) -> String {
    classes
        .iter()
        .map(|&c| match c {
            NEG => "n ".to_string(),
            POS => "p ".to_string(),
            _ => ". ".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}
