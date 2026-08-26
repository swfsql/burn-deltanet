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
///   S⁻ = α ⊙ S          (α = exp(g); skipped when `g_bh` is `None`)
///   u  = β (v − S⁻ᵀ k)
///   S  = S⁻ + k uᵀ
///   y  = Sᵀ (scale · q)
/// ```
///
/// `q`/`k` are expected already activated and normalised; `scale` is applied
/// here so the caller does not have to.
///
/// # Shapes
/// - `q_bhk`, `k_bhk`: `[batch, nheads, head_k_dim]`
/// - `v_bhv`: `[batch, nheads, head_v_dim]`
/// - `beta_bh`, `g_bh`: `[batch, nheads]`
/// - `state_bhkv`: `[batch, nheads, head_k_dim, head_v_dim]`
/// - returns `(y_bhv, next_state_bhkv)`
pub fn delta_step(
    q_bhk: Tensor<3>,
    k_bhk: Tensor<3>,
    v_bhv: Tensor<3>,
    beta_bh: Tensor<2>,
    g_bh: Option<Tensor<2>>,
    state_bhkv: Tensor<4>,
    scale: f64,
) -> (Tensor<3>, Tensor<4>) {
    let [batch, nheads, head_k_dim] = q_bhk.dims();
    let [_b, _h, head_v_dim] = v_bhv.dims();
    assert_eq!([batch, nheads, head_k_dim], k_bhk.dims());
    assert_eq!([batch, nheads], beta_bh.dims());
    assert_eq!([batch, nheads, head_k_dim, head_v_dim], state_bhkv.dims());

    // ── Forget gate: S⁻ = exp(g) · S ─────────────────────────────────────────
    let state_bhkv = match g_bh {
        Some(g_bh) => {
            assert_eq!([batch, nheads], g_bh.dims());
            let decay_bh11: Tensor<4> = g_bh.exp().unsqueeze_dims(&[-1, -1]);
            assert_eq!([batch, nheads, 1, 1], decay_bh11.dims());
            state_bhkv * decay_bh11
        }
        None => state_bhkv,
    };

    // ── Retrieve: what the state currently associates with k ────────────────
    // A `[1, k] @ [k, v]` matmul rather than a broadcast-multiply-and-sum, so
    // the backend sees a GEMM.
    let retrieved_bhv = k_bhk
        .clone()
        .unsqueeze_dim::<4>(2) // [batch, nheads, 1, head_k_dim]
        .matmul(state_bhkv.clone())
        .squeeze_dim(2);
    assert_eq!([batch, nheads, head_v_dim], retrieved_bhv.dims());

    // ── The delta: u = β (v − retrieved) ────────────────────────────────────
    let u_bhv = (v_bhv - retrieved_bhv) * beta_bh.unsqueeze_dim::<3>(2);
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
    pub fn delta_recurrent(self) -> (Tensor<4>, Tensor<4>) {
        let (batch, sequence, nheads, _head_k_dim, head_v_dim) = self.dims();
        let scale = self.resolved_scale();
        let DeltaInput {
            q_bshk,
            k_bshk,
            v_bshv,
            beta_bsh,
            g_bsh,
            state_bhkv,
            scale: _,
        } = self;

        let mut state_bhkv = state_bhkv;
        let mut ys = Vec::with_capacity(sequence);
        for t in 0..sequence {
            let q_bhk = q_bshk.clone().narrow(1, t, 1).squeeze_dim(1);
            let k_bhk = k_bshk.clone().narrow(1, t, 1).squeeze_dim(1);
            let v_bhv = v_bshv.clone().narrow(1, t, 1).squeeze_dim(1);
            let beta_bh = beta_bsh.clone().narrow(1, t, 1).squeeze_dim(1);
            let g_bh = g_bsh
                .as_ref()
                .map(|g| g.clone().narrow(1, t, 1).squeeze_dim(1));

            let (y_bhv, next) = delta_step(q_bhk, k_bhk, v_bhv, beta_bh, g_bh, state_bhkv, scale);
            state_bhkv = next;
            ys.push(y_bhv.unsqueeze_dim(1));
        }

        let y_bshv = Tensor::cat(ys, 1);
        assert_eq!([batch, sequence, nheads, head_v_dim], y_bshv.dims());
        (y_bshv, state_bhkv)
    }
}
