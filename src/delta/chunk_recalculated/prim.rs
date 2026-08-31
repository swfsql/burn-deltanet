//! The handful of primitive ops the chunk body needs beyond [`F`]'s own API.
//!
//! [`F`] mirrors the slice of the high-level `Tensor` API that a recompute
//! backward generally needs; the delta rule wants three more. They are
//! `B::float_*` calls of exactly the shape of [`F`]'s own methods and carry no
//! delta-specific meaning — an extension trait here rather than a fork of the
//! wrapper (which lives in `burn-stack`, and is pinned by revision).

use burn::backend::{Backend, Scalar, get_device_settings};
use burn_stack::utils::fprim::{F, Mask};

/// Ops on [`F`] that the chunk body needs and [`F`] does not carry.
pub(crate) trait FPrimExt<B: Backend, const D: usize>: Sized {
    /// Element-wise multiplication by a scalar.
    fn mul_scalar(self, value: f64) -> Self;

    /// Element-wise ceiling at `max`.
    fn clamp_max(self, max: f64) -> Self;

    /// Zero everything strictly above the `diagonal` — the mirror of
    /// [`F::triu`], and the same `mask_fill` it is.
    fn tril(self, diagonal: i64) -> Self;

    /// Mask that is `true` wherever `self >= value`: exactly where
    /// [`Self::clamp_max`] at `value` binds, so a clamped value's gradient is
    /// `d.mask_fill(x.ge_elem(value), 0.0)`.
    fn ge_elem(&self, value: f64) -> Mask<B>;
}

impl<B: Backend, const D: usize> FPrimExt<B, D> for F<B, D> {
    fn mul_scalar(self, value: f64) -> Self {
        F::new(B::float_mul_scalar(self.inner(), Scalar::from(value)))
    }

    fn clamp_max(self, max: f64) -> Self {
        F::new(B::float_clamp_max(self.inner(), Scalar::from(max)))
    }

    fn tril(self, diagonal: i64) -> Self {
        let dims = self.dims();
        let device = self.device();
        self.mask_fill(tril_mask_like::<B, D>(dims, diagonal, &device), 0.0)
    }

    fn ge_elem(&self, value: f64) -> Mask<B> {
        let device = self.device();
        let bool_dtype = get_device_settings::<B>(&device).bool_dtype;
        Mask(B::float_greater_equal_elem(
            self.clone().inner(),
            Scalar::from(value),
            bool_dtype,
        ))
    }
}

/// The mask a `tril(diagonal)` over the last two axes of `shape` fills:
/// `true` strictly above the diagonal, broadcast to the full rank.
pub(crate) fn tril_mask_like<B: Backend, const D: usize>(
    shape: [usize; D],
    diagonal: i64,
    device: &burn::backend::tensor::Device<B>,
) -> Mask<B> {
    let mut lead = [1usize; D];
    lead[D - 2] = shape[D - 2];
    lead[D - 1] = shape[D - 1];
    Mask::tril_mask(shape[D - 2], shape[D - 1], diagonal, device)
        .reshape(lead)
        .expand(shape)
}
