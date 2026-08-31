use super::*;
use burn::module::Param;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

/// A batched strictly-lower-triangular `N` with uncorrelated entries — the
/// easy case, where even the Neumann series stays well-conditioned.
fn random_strict_lower(batch: usize, size: usize, scale: f64, device: &Device) -> Tensor<3> {
    Tensor::<3>::random(
        [batch, size, size],
        Distribution::Normal(0.0, scale),
        device,
    )
    .tril(-1)
}

/// `(I − N) T` must be the identity for both solves — that is the definition,
/// independent of how the series was summed.
fn check_inverse(size: usize, scale: f64, solve: TriSolve, tol: f32) {
    let device: Device = Default::default();
    let batch = 3;
    let n = random_strict_lower(batch, size, scale, &device);
    let t = unit_lower_inverse(n.clone(), solve);

    let identity: Tensor<3> = Tensor::eye(size, &device).unsqueeze();
    let product = (identity.clone() - n).matmul(t);
    let diff = max_abs_diff(product, identity);
    assert!(
        diff < tol,
        "(I - N) T != I for size {size} / {solve:?}: max abs diff {diff}",
    );
}

#[test]
fn blocked_inverts() {
    for size in [1, 2, 3, 4, 5, 7, 8, 16, 32, 64, 128] {
        check_inverse(size, 0.3, TriSolve::Blocked, 1e-3);
    }
}

#[test]
fn neumann_inverts() {
    for size in [1, 2, 3, 4, 5, 7, 8, 16, 32] {
        check_inverse(size, 0.3, TriSolve::Neumann, 1e-3);
    }
}

/// Where the series is still conditioned the two solves are the same function,
/// so they must agree on values and on gradients.
///
/// The `tril(-1)` sits *inside* the graph, exactly as the chunk path builds
/// `N`: only the strict lower triangle is an input, and the two solves disagree
/// about the (unused) rest — `Neumann` multiplies by the whole matrix.
#[test]
fn blocked_matches_neumann_values_and_grads() {
    let device: Device = Default::default();
    let (batch, size) = (2, 16);
    let n_inner = random_strict_lower(batch, size, 0.3, &device);
    let head = Tensor::<3>::random([batch, size, size], Distribution::Normal(0.0, 1.0), &device);

    let run = |solve: TriSolve| {
        let n = Param::from_tensor(Tensor::from_inner(n_inner.clone()));
        let t = unit_lower_inverse(n.val().tril(-1), solve);
        let t_inner = t.clone().inner();
        let loss = (t * Tensor::from_inner(head.clone())).sum();
        let grads = loss.backward();
        (t_inner, n.val().grad(&grads).expect("grad n"))
    };

    let (t_blocked, d_blocked) = run(TriSolve::Blocked);
    let (t_neumann, d_neumann) = run(TriSolve::Neumann);

    let value_diff = max_abs_diff(t_blocked, t_neumann);
    let grad_diff = max_abs_diff(d_blocked, d_neumann);
    assert!(value_diff < 1e-4, "value mismatch: {value_diff}");
    assert!(grad_diff < 1e-4, "grad mismatch: {grad_diff}");
}

/// The strictly-lower structure makes `T` unit lower triangular: `T = I + N +
/// N² + …` adds nothing on or above the diagonal beyond the identity.
#[test]
fn inverse_is_unit_lower_triangular() {
    let device: Device = Default::default();
    let size = 8;
    let n = random_strict_lower(2, size, 0.5, &device);
    let t = unit_lower_inverse(n, TriSolve::Blocked);

    let identity: Tensor<3> = Tensor::eye(size, &device).unsqueeze();
    let strict_upper = t.clone().triu(1);
    assert!(
        strict_upper.abs().max().into_scalar::<f32>() < 1e-6,
        "T has a non-zero strict upper triangle",
    );
    let diagonal_diff = max_abs_diff(t.tril(0).triu(0), identity.tril(0).triu(0));
    assert!(diagonal_diff < 1e-6, "T's diagonal is not 1: {diagonal_diff}");
}

/// The case that motivates [`TriSolve::Blocked`]: every key in the chunk points
/// the same way, so `N` is `−β` on the whole strict lower triangle. `I − N` is
/// then the (scaled) prefix-sum matrix — its inverse differences neighbours and
/// holds nothing bigger than `β` — while the series that sums to it passes
/// through terms of order `C(L−2, L/2)`.
#[test]
fn blocked_inverts_a_constant_key_chunk() {
    let device: Device = Default::default();
    for size in [8, 16, 32, 64, 128] {
        for beta in [0.9, 1.0, 1.9] {
            let n = Tensor::<2>::full([size, size], -beta, &device).tril(-1);
            let t = unit_lower_inverse(n.clone(), TriSolve::Blocked);
            let identity = Tensor::<2>::eye(size, &device);
            let diff = max_abs_diff((identity.clone() - n).matmul(t.clone()), identity);
            assert!(
                diff < 1e-4,
                "(I - N) T != I for size {size} / beta {beta}: max abs diff {diff}",
            );
            // `T` is unit lower triangular and, here, holds nothing bigger
            // than `max(1, β)` at any length — the property the series loses.
            let magnitude = t.abs().max().into_scalar::<f32>();
            let bound = (beta as f32).max(1.0) + 1e-4;
            assert!(
                magnitude <= bound,
                "T grew past {bound} for size {size} / beta {beta}: {magnitude}",
            );
        }
    }
}
