use super::*;
use burn::module::Param;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

/// A batched strictly-lower-triangular `N` with entries small enough that the
/// finite Neumann series stays well-conditioned (the delta rule's own `N` is
/// bounded the same way: `‖k‖ = 1` and `β ≤ 2`).
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
fn doubling_inverts() {
    for size in [1, 2, 3, 4, 5, 7, 8, 16, 32, 64] {
        check_inverse(size, 0.3, TriSolve::Doubling, 1e-3);
    }
}

#[test]
fn neumann_inverts() {
    for size in [1, 2, 3, 4, 5, 7, 8, 16, 32] {
        check_inverse(size, 0.3, TriSolve::Neumann, 1e-3);
    }
}

/// The two solves evaluate the same finite polynomial, so they must agree on
/// values and on gradients (they differ only in how the sum is factored).
#[test]
fn doubling_matches_neumann_values_and_grads() {
    let device: Device = Default::default();
    let (batch, size) = (2, 16);
    let n_inner = random_strict_lower(batch, size, 0.3, &device);
    let head = Tensor::<3>::random([batch, size, size], Distribution::Normal(0.0, 1.0), &device);

    let run = |solve: TriSolve| {
        let n = Param::from_tensor(Tensor::from_inner(n_inner.clone()));
        let t = unit_lower_inverse(n.val(), solve);
        let t_inner = t.clone().inner();
        let loss = (t * Tensor::from_inner(head.clone())).sum();
        let grads = loss.backward();
        (t_inner, n.val().grad(&grads).expect("grad n"))
    };

    let (t_doubling, d_doubling) = run(TriSolve::Doubling);
    let (t_neumann, d_neumann) = run(TriSolve::Neumann);

    let value_diff = max_abs_diff(t_doubling, t_neumann);
    let grad_diff = max_abs_diff(d_doubling, d_neumann);
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
    let t = unit_lower_inverse(n, TriSolve::Doubling);

    let identity: Tensor<3> = Tensor::eye(size, &device).unsqueeze();
    let strict_upper = t.clone().triu(1);
    assert!(
        strict_upper.abs().max().into_scalar::<f32>() < 1e-6,
        "T has a non-zero strict upper triangle",
    );
    let diagonal_diff = max_abs_diff(t.tril(0).triu(0), identity.tril(0).triu(0));
    assert!(diagonal_diff < 1e-6, "T's diagonal is not 1: {diagonal_diff}");
}
