"""The implementation closely follows that of https://github.com/apple/ml-calibration/blob/main/src/relplot/kernels.py."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def smooth_round_to_grid(f: Tensor, y: Tensor, eval_points: int) -> Tensor:
    """Discretizes continuous positions into a fixed-size grid using linear weight distribution.

    This function acts as a "soft" histogram. Each value in `y` at position `f` is
    distributed between the two nearest integer bins in the grid based on its
    fractional distance, ensuring the operation is differentiable with respect to `f`.

    Args:
        f: A 1D Tensor of positions, expected to be normalized in the range [0, 1].
        y: A 1D Tensor of weights or values associated with each position in `f`.
        eval_points: The number of bins in the output grid.

    Returns:
        A Tensor of shape `(eval_points,)` containing the aggregated weights.
    """
    values = torch.zeros(eval_points, device=f.device)

    scaled_f = f * (eval_points - 1)
    # Clamp to ensure bins+1 is always valid, exactly as in np.clip
    bins = scaled_f.long().clamp(0, eval_points - 2)
    frac = scaled_f - bins

    # Efficiently scatter the weights into the grid
    values.index_add_(0, bins, (1 - frac) * y)
    values.index_add_(0, bins + 1, frac * y)
    return values


def interpolate(t: Tensor, y: Tensor) -> Tensor:
    """Performs linear interpolation to map values from a discrete grid back to continuous points.

    Args:
        t: A 1D Tensor of query positions in the range [0, 1].
        y: A 1D Tensor representing the discrete grid of values.

    Returns:
        The interpolated values at positions `t`.
    """
    num_buckets = y.size(0)
    bucket_size = 1.0 / (num_buckets - 1)

    scaled_t = t * (num_buckets - 1)
    inds = scaled_t.long().clamp(0, num_buckets - 2)
    residual = (t - inds * bucket_size) / bucket_size
    return y[inds] * (1 - residual) + y[inds + 1] * residual


class GaussianKernel:
    r"""Standard Gaussian (RBF) Kernel smoother using 1D convolution.

    The kernel is defined as:
    $$K(d) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{d^2}{2\\sigma^2}\\right)$$
    """

    def __init__(self, sigma: float) -> None:
        """Args:
        sigma: The standard deviation (bandwidth) of the Gaussian kernel.
        """
        self.sigma = sigma

    def kernel_ev(self, num_eval_points: int, device: torch.device) -> Tensor:
        """Generates a discrete Gaussian kernel window centered at 0.5."""
        t = torch.linspace(0, 1, num_eval_points, device=device)
        res = torch.exp(-((t - 0.5) ** 2) / (2 * self.sigma**2))
        res /= math.sqrt(2 * math.pi) * self.sigma
        return res

    def convolve(self, values: Tensor, eval_points: int) -> Tensor:
        """Performs 1D convolution with 'same' padding to smooth the grid."""
        ker = self.kernel_ev(eval_points, values.device).view(1, 1, -1)
        v = values.view(1, 1, -1)
        # 1D Convolution with 'same' padding
        smoothed = F.conv1d(v, ker, padding="same")
        return smoothed.view(-1)

    def apply(self, f: Tensor, y: Tensor, x_eval: Tensor, eval_points: int | None = None) -> Tensor:
        """Maps data to a grid, convolves with the Gaussian, and interpolates back."""
        if eval_points is None:
            eval_points = max(2000, round(20 / self.sigma))
        eval_points = (eval_points // 2) + 1

        values = smooth_round_to_grid(f, y, eval_points)
        smoothed = self.convolve(values, eval_points)
        return interpolate(x_eval, smoothed)

    def smooth(
        self, f: Tensor, y: Tensor, x_eval: Tensor, eps: float = 1e-4
    ) -> tuple[Tensor, Tensor]:
        """Performs kernel smoothing.

        Args:
            f: Input positions [0, 1].
            y: Input values.
            x_eval: Points at which to evaluate the smoothed function.
            eps: A small constant to prevent division by zero in low-density regions.

        Returns:
            A tuple of (smoothed_values, density_estimates).
        """
        ys = self.apply(f, y, x_eval)
        dens = self.apply(f, torch.ones_like(y), x_eval) + eps
        return ys / dens, dens


class ReflectedGaussianKernel(GaussianKernel):
    """Gaussian Kernel with boundary reflection to mitigate edge bias.

    Standard kernels often exhibit 'boundary drop-off' where the density
    appears lower near 0 and 1. This class pads the input by reflecting
    the signal across the boundaries before convolving.
    """

    def convolve(self, values: Tensor, eval_points: int) -> Tensor:
        ker = self.kernel_ev(eval_points, values.device).view(1, 1, -1)

        # Reflect boundaries
        flip_v = torch.flip(values, dims=[0])
        ext_vals = torch.cat([flip_v[:-1], values, flip_v[1:]], dim=0).view(1, 1, -1)

        smoothed = F.conv1d(ext_vals, ker, padding="valid").view(-1)
        start = eval_points // 2
        return smoothed[start : start + eval_points]


class LogitGaussianKernel:
    r"""Kernel smoother that operates in logit space.

    Useful for data strictly bounded in [0, 1] (like probabilities). It transforms
    data using the logit function $L(x) = \\ln(x / (1-x))$, performs Gaussian
    smoothing in the unconstrained space, and maps back.
    """

    def __init__(self, sigma: float):
        """Args:
        sigma: Bandwidth applied in the logit-transformed space.
        """
        self.sigma = sigma

    def transform(
        self, f: Tensor, x_eval: Tensor, eps: float = 0.001
    ) -> tuple[Tensor, Tensor, float]:
        """Maps [0, 1] data to a normalized logit space."""
        z = x_eval.view(-1).double()[1:-1]
        logit_z = torch.log(z / (1 - z))

        f_clamped = torch.clamp(f, eps, 1 - eps)
        logit_f = torch.log(f_clamped / (1 - f_clamped))

        range_min = min(logit_f.min().item(), logit_z.min().item())
        range_max = max(logit_f.max().item(), logit_z.max().item())
        span = range_max - range_min if range_max > range_min else 1e-9

        logit_f = (logit_f - range_min) / span
        logit_z = (logit_z - range_min) / span
        return logit_f, logit_z, 4 / span

    def smooth(self, f: Tensor, y: Tensor, x_eval: Tensor) -> tuple[Tensor, Tensor]:
        """Performs smoothing in logit space with a Jacobian correction for the density.

        The density is adjusted by $1 / (x(1-x))$ to account for the stretching
        of the logit transform.
        """
        logit_f, logit_z, scale = self.transform(f, x_eval)
        kernel = GaussianKernel(self.sigma * scale)
        ys, dens = kernel.smooth(logit_f, y, logit_z)

        dens /= x_eval[1:-1] * (1 - x_eval[1:-1])
        dens *= len(f) * len(x_eval) / (dens.sum() + 1e-9)

        # Re-pad the ends that were sliced during the logit transform
        ys_full = torch.cat([ys[:1], ys, ys[-1:]])
        dens_full = torch.cat([dens[:1], dens, dens[-1:]])

        return ys_full, dens_full
