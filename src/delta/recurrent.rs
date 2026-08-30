//! The token-by-token delta-rule recurrence.
//!
//! [`delta_step`] is one tick of the definition in the [module
//! header](crate::delta) — the primitive every family's `step()` decodes with,
//! and, unrolled by [`DeltaInput::delta_recurrent`], the reference the
//! [chunked](super::chunk) path is checked against.
//!
//! Cost per token is `O(nheads · head_k_dim · head_v_dim)` — the state size —
//! with no growing KV cache, which is the property the whole family is for.

use burn::prelude::*;
use burn_stack::modules::sanity as san;

use super::path::DeltaInput;

/// One delta-rule tick, for all heads of a batch at once.
///
/// ```text
///   S⁻ = α ⊙ S              (α = exp(g), on the key axis; skipped when `None`)
///   u  = write ⊙ v − S⁻ᵀ (erase ⊙ k)
///   S  = S⁻ + k uᵀ
///   y  = Sᵀ (scale · q)
/// ```
///
/// With a per-head `β` — `erase = write = β`, both of width 1 — that reads
/// `u = β (v − S⁻ᵀ k)`, the scalar delta rule verbatim. Widening the gates onto
/// their channel axes decouples the two halves of the write, which is what
/// [GDN-2](crate::gated_deltanet_2) is.
///
/// `q`/`k` are expected already activated and normalised; `scale` is applied
/// here so the caller does not have to.
///
/// # Shapes
/// - `q_bhk`, `k_bhk`: `[batch, nheads, head_k_dim]`
/// - `v_bhv`: `[batch, nheads, head_v_dim]`
/// - `erase_bhK`, `g_bhK`: `[batch, nheads, 1 | head_k_dim]`
/// - `write_bhV`: `[batch, nheads, 1 | head_v_dim]`
/// - `state_bhkv`: `[batch, nheads, head_k_dim, head_v_dim]`
/// - returns `(y_bhv, next_state_bhkv)`
#[allow(non_snake_case)]
pub fn delta_step(
    q_bhk: Tensor<3>,
    k_bhk: Tensor<3>,
    v_bhv: Tensor<3>,
    erase_bhK: Tensor<3>,
    write_bhV: Tensor<3>,
    g_bhK: Option<Tensor<3>>,
    state_bhkv: Tensor<4>,
    scale: f64,
) -> (Tensor<3>, Tensor<4>) {
    let [batch, nheads, head_k_dim] = q_bhk.dims();
    let [_b, _h, head_v_dim] = v_bhv.dims();
    assert_eq!([batch, nheads, head_k_dim], k_bhk.dims());
    assert_eq!([batch, nheads, head_k_dim, head_v_dim], state_bhkv.dims());

    // ── Forget gate: S⁻ = exp(g) ⊙ S, along the state's key axis ────────────
    let state_bhkv = match g_bhK {
        Some(g_bhK) => {
            // `[batch, nheads, 1 | head_k_dim, 1]`: broadcasts over the values.
            let decay_bhK1: Tensor<4> = g_bhK.exp().unsqueeze_dim(3);
            state_bhkv * decay_bhK1
        }
        None => state_bhkv,
    };

    // ── Retrieve: what the state currently associates with the erase key ────
    // A `[1, k] @ [k, v]` matmul rather than a broadcast-multiply-and-sum, so
    // the backend sees a GEMM.
    let erased_bhv = (k_bhk.clone() * erase_bhK)
        .unsqueeze_dim::<4>(2) // [batch, nheads, 1, head_k_dim]
        .matmul(state_bhkv.clone())
        .squeeze_dim(2);
    assert_eq!([batch, nheads, head_v_dim], erased_bhv.dims());

    // ── The delta: what is to be written, minus what is already there ───────
    let u_bhv = v_bhv * write_bhV - erased_bhv;
    assert_eq!([batch, nheads, head_v_dim], u_bhv.dims());

    // ── Write: the rank-1 correction S ← S⁻ + k uᵀ ──────────────────────────
    let write_bhkv = k_bhk
        .unsqueeze_dim::<4>(3) // [batch, nheads, head_k_dim, 1]
        .matmul(u_bhv.unsqueeze_dim::<4>(2)); // [batch, nheads, 1, head_v_dim]
    let next_state_bhkv = state_bhkv + write_bhkv;
    assert_eq!(
        [batch, nheads, head_k_dim, head_v_dim],
        next_state_bhkv.dims()
    );

    // ── Read out against the *updated* state ────────────────────────────────
    let y_bhv = (q_bhk * scale)
        .unsqueeze_dim::<4>(2)
        .matmul(next_state_bhkv.clone())
        .squeeze_dim(2);
    assert_eq!([batch, nheads, head_v_dim], y_bhv.dims());

    san(&y_bhv);
    san(&next_state_bhkv);
    (y_bhv, next_state_bhkv)
}

impl DeltaInput {
    /// Unroll [`delta_step`] over the sequence.
    ///
    /// `O(sequence)` serial ticks — the definition, and the right choice for a
    /// short prefill where the chunk algebra would not pay for itself.
    #[allow(non_snake_case)]
    pub fn delta_recurrent(self) -> (Tensor<4>, Tensor<4>) {
        let (batch, sequence, nheads, _head_k_dim, head_v_dim) = self.dims();
        let scale = self.resolved_scale();
        let DeltaInput {
            q_bshk,
            k_bshk,
            v_bshv,
            erase_bshK,
            write_bshV,
            g_bshK,
            state_bhkv,
            scale: _,
        } = self;

        let mut state_bhkv = state_bhkv;
        let mut ys = Vec::with_capacity(sequence);
        for t in 0..sequence {
            let at = |x_bs: &Tensor<4>| -> Tensor<3> { x_bs.clone().narrow(1, t, 1).squeeze_dim(1) };
            let (y_bhv, next) = delta_step(
                at(&q_bshk),
                at(&k_bshk),
                at(&v_bshv),
                at(&erase_bshK),
                at(&write_bshV),
                g_bshK.as_ref().map(at),
                state_bhkv,
                scale,
            );
            state_bhkv = next;
            ys.push(y_bhv.unsqueeze_dim(1));
        }

        let y_bshv = Tensor::cat(ys, 1);
        assert_eq!([batch, sequence, nheads, head_v_dim], y_bshv.dims());
        (y_bshv, state_bhkv)
    }
}
