//! The fused causal depthwise short convolution, plus its rolling window.
//!
//! A depthwise convolution treats every channel independently, so the three
//! per-projection `ShortConvolution`s of the reference implementation (one each
//! for `q`, `k`, `v`) are exactly one depthwise convolution over the
//! concatenated `[q | k | v]` channel axis — one weight tensor, one kernel
//! launch, one cache tensor. That is the form kept here, mirroring how Mamba-2
//! convolves its fused `xbc`.
//!
//! Causality is manual: the convolution is configured with `Valid` padding and
//! the input is left-padded with the previous call's tail, taken from the
//! rolling window. The window carries `conv_kernel` columns — one more than the
//! `conv_kernel − 1` a `forward` needs as its pad — because a single-token
//! [`ShortConv::step`] evaluates the filter against the whole window.
//!
//! # Shapes
//! - window: `[batch, conv_dim, conv_kernel]`
//! - `forward`: `[batch, sequence, conv_dim] → [batch, sequence, conv_dim]`
//! - `step`: `[batch, conv_dim] → [batch, conv_dim]`

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Initializer, PaddingConfig1d};
use burn::prelude::*;
use burn_stack::modules::{Silu, sanity as san};

/// The element-wise activation folded into the short convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ConvActivation {
    /// `x · σ(x)` — what every family in this crate uses.
    #[default]
    Silu,
    /// No activation.
    Identity,
}

impl ConvActivation {
    /// Apply the activation element-wise.
    pub fn apply<const D: usize>(self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Silu => Silu::new().forward(x),
            Self::Identity => x,
        }
    }
}

/// A causal depthwise `Conv1d` over `conv_dim` channels with a fused activation.
#[derive(Module, Debug)]
pub struct ShortConv {
    /// The depthwise convolution: `groups = conv_dim`, `Valid` padding.
    pub conv1d: Conv1d,
    /// Activation applied to the convolution output. A non-parameter constant —
    /// `#[module(skip)]` keeps it out of the record and carries it through
    /// `load_record`/`to_device`/… unchanged.
    #[module(skip)]
    pub activation: ConvActivation,
}

impl ShortConv {
    /// Number of channels (= groups) the convolution runs over.
    pub fn conv_dim(&self) -> usize {
        let [conv_dim, _one, _kernel] = self.conv1d.weight.dims();
        conv_dim
    }

    /// The convolution window length.
    pub fn conv_kernel(&self) -> usize {
        let [_conv_dim, _one, kernel] = self.conv1d.weight.dims();
        kernel
    }

    /// A zero rolling window, i.e. "no previous tokens".
    pub fn zero_window(&self, batch: usize, device: &Device) -> Tensor<3> {
        Tensor::zeros(
            Shape::new([batch, self.conv_dim(), self.conv_kernel()]),
            device,
        )
    }

    /// Full-sequence causal convolution.
    ///
    /// Returns the activated output and the updated rolling window (the last
    /// `conv_kernel` columns of the causally-padded input).
    #[allow(non_snake_case)]
    pub fn forward(&self, x_bsw: Tensor<3>, window_bwc: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let [batch, sequence, conv_dim] = x_bsw.dims();
        let conv_kernel = self.conv_kernel();
        assert_eq!(conv_dim, self.conv_dim());
        assert_eq!([batch, conv_dim, conv_kernel], window_bwc.dims());
        assert!(sequence > 0, "sequence length must be at least 1");
        san(&x_bsw);

        let x_bws = x_bsw.transpose();

        // Left-pad with the previous call's tail: the window's newest
        // `conv_kernel - 1` columns.
        let padded_bwS = if conv_kernel >= 2 {
            let tail_bwC = window_bwc.slice(s![.., .., 1..]);
            assert_eq!([batch, conv_dim, conv_kernel - 1], tail_bwC.dims());
            Tensor::cat(vec![tail_bwC, x_bws], 2)
        } else {
            x_bws
        };
        assert_eq!(
            [batch, conv_dim, (conv_kernel - 1) + sequence],
            padded_bwS.dims()
        );

        // The next window is the last `conv_kernel` columns of the padded input.
        let next_window_bwc = padded_bwS.clone().slice(s![.., .., (sequence - 1)..]);
        assert_eq!([batch, conv_dim, conv_kernel], next_window_bwc.dims());

        let y_bws = self.conv1d.forward(padded_bwS);
        assert_eq!([batch, conv_dim, sequence], y_bws.dims());

        let y_bsw = self.activation.apply(y_bws.transpose());
        san(&y_bsw);
        (y_bsw, next_window_bwc)
    }

    /// Single-token causal convolution: slide the window and evaluate the
    /// filter against it.
    #[allow(non_snake_case)]
    pub fn step(&self, x_bw: Tensor<2>, window_bwc: Tensor<3>) -> (Tensor<2>, Tensor<3>) {
        let [batch, conv_dim] = x_bw.dims();
        let conv_kernel = self.conv_kernel();
        assert_eq!(conv_dim, self.conv_dim());
        assert_eq!([batch, conv_dim, conv_kernel], window_bwc.dims());

        // Drop the oldest column, append the new token as the newest one.
        let next_window_bwc = if conv_kernel >= 2 {
            let tail_bwC = window_bwc.slice(s![.., .., 1..]);
            Tensor::cat(vec![tail_bwC, x_bw.unsqueeze_dim(2)], 2)
        } else {
            x_bw.unsqueeze_dim(2)
        };
        assert_eq!([batch, conv_dim, conv_kernel], next_window_bwc.dims());

        // One step of a depthwise convolution is a dot product of the window
        // with the filter along the kernel axis.
        let weight_bwc = self
            .conv1d
            .weight
            .val() // [conv_dim, 1, conv_kernel]
            .swap_dims(0, 1)
            .expand([batch, conv_dim, conv_kernel]);
        let mut y_bw = (next_window_bwc.clone() * weight_bwc).sum_dim(2).squeeze_dim(2);
        assert_eq!([batch, conv_dim], y_bw.dims());

        if let Some(bias_w) = &self.conv1d.bias {
            y_bw = y_bw + bias_w.val().unsqueeze();
        }

        let y_bw = self.activation.apply(y_bw);
        san(&y_bw);
        (y_bw, next_window_bwc)
    }
}

/// Configuration for [`ShortConv`].
#[derive(Config, Debug)]
pub struct ShortConvConfig {
    /// Number of (independent) channels.
    pub conv_dim: usize,
    /// Window length. The reference uses 4.
    #[config(default = 4)]
    pub conv_kernel: usize,
    /// Whether the convolution carries a bias.
    #[config(default = false)]
    pub has_bias: bool,
    /// Activation folded into the convolution output.
    #[config(default = "ConvActivation::Silu")]
    pub activation: ConvActivation,
}

impl ShortConvConfig {
    /// Allocate the convolution on `device`.
    pub fn init(&self, device: &Device) -> ShortConv {
        assert!(self.conv_kernel > 0, "conv_kernel must be at least 1");
        // PyTorch's default `Conv1d` initialiser: U(-1/√fan_in, 1/√fan_in) with
        // `fan_in = in_channels / groups * kernel_size = conv_kernel`.
        let bound = 1.0 / (self.conv_kernel as f64).sqrt();
        let conv1d = Conv1dConfig::new(self.conv_dim, self.conv_dim, self.conv_kernel)
            .with_padding(PaddingConfig1d::Valid)
            .with_groups(self.conv_dim)
            .with_bias(self.has_bias)
            .with_initializer(Initializer::Uniform {
                min: -bound,
                max: bound,
            })
            .init(device);
        ShortConv {
            conv1d,
            activation: self.activation,
        }
    }
}
