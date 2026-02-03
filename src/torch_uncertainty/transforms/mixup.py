from abc import ABC, abstractmethod
from importlib import util
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup as timm_Mixup
from torch import Tensor, nn

from torch_uncertainty.layers import Identity

if util.find_spec("scipy"):
    import scipy

    scipy_installed = True
else:  # coverage: ignore
    scipy_installed = False


MIXUP_PARAMS = {
    "mixtype": None,
    "isobatch": False,
    "kernel_tau_max": 1.0,
    "kernel_tau_std": 0.5,
    "mixup_alpha": 0,
    "cutmix_alpha": 0,
    "dist_sim": "emb",
}


def beta_warping(x, alpha_cdf: float = 1.0, eps: float = 1e-12) -> float:
    return scipy.stats.beta.cdf(x, a=alpha_cdf + eps, b=alpha_cdf + eps)


def sim_gauss_kernel(dist, tau_max: float = 1.0, tau_std: float = 0.5) -> float:
    dist_rate = tau_max * np.exp(-(dist - 1) / (np.mean(dist) * 2 * tau_std * tau_std))
    return 1 / (dist_rate + 1e-12)


# ruff: noqa: ERA001
# def tensor_linspace(start: Tensor, stop: Tensor, num: int):
#     """
#     Creates a tensor of shape [num, *start.shape] whose values are evenly
#     spaced from start to end, inclusive.
#     Replicates but the multi-dimensional behaviour of numpy.linspace in PyTorch.
#     """
#     # create a tensor of 'num' steps from 0 to 1
#     steps = torch.arange(num, dtype=torch.float32, device=start.device) / (
#         num - 1
#     )

#     # reshape the 'steps' tensor to [-1, *([1]*start.ndim)]
#     # to allow for broadcastings
#     # using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here
#     # but torchscript
#     # "cannot statically infer the expected size of a list in this context",
#     # hence the code below
#     for i in range(start.ndim):
#         steps = steps.unsqueeze(-1)

#     # the output starts at 'start' and increments until 'stop' in each dimension
#     out = start[None] + steps * (stop - start)[None]

#     return out


# def torch_beta_cdf(
#     x: Tensor, c1: Tensor | float, c2: Tensor | float, npts=100, eps=1e-12
# ):
#     if isinstance(c1, float):
#         if c1 == c2:
#             c1 = Tensor([c1], device=x.device)
#             c2 = c1
#         else:
#             c1 = Tensor([c1], device=x.device)
#     if isinstance(c2, float):
#         c2 = Tensor([c2], device=x.device)
#     bt = torch.distributions.Beta(c1, c2)

#     if isinstance(x, float):
#         x = Tensor(x)

#     X = tensor_linspace(torch.zeros_like(x) + eps, x, npts)
#     return torch.trapezoid(bt.log_prob(X).exp(), X, dim=0)


# def torch_beta_warping(
#     x: Tensor, alpha_cdf: float | Tensor = 1.0, eps=1e-12, npts=100
# ):
#     return torch_beta_cdf(
#         x=x, c1=alpha_cdf + eps, c2=alpha_cdf + eps, npts=npts, eps=eps
#     )


# def torch_sim_gauss_kernel(dist: Tensor, tau_max=1.0, tau_std=0.5):
#     dist_rate = tau_max * torch.exp(
#         -(dist - 1) / (torch.mean(dist) * 2 * tau_std * tau_std)
#     )

#     return 1 / (dist_rate + 1e-12)


# TODO: Should be a torchvision transform
class AbstractMixup(ABC, nn.Module):
    def __init__(
        self, alpha: float, num_classes: int, isobatch: bool = False, **kwargs: Any
    ) -> None:
        """Abstract Mixup class.

        Args:
            alpha (float, optional): Mixup alpha.
            num_classes (int): Number of classes.
            isobatch (bool, optional): Whether to use a single coefficient for the whole batch
                instead of for each pair. Defaults to ``False``.
            **kwargs (Any): Keyword arguments for compatibility.
        """
        super().__init__()
        self.alpha = alpha
        self.distribution = torch.distributions.Beta(alpha, alpha)
        self.num_classes = num_classes
        self.isobatch = isobatch

    def _get_params(self, batch_size: int, device: torch.device) -> tuple[float, Tensor]:
        if self.isobatch:
            lam = self.distribution.sample()
        else:
            lam = self.distribution.sample((batch_size,)).to(device=device)
        index = torch.randperm(batch_size, device=device)
        return lam, index

    def _linear_mixing(
        self,
        lam: Tensor | float,
        inp: Tensor,
        index: Tensor,
    ) -> Tensor:
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1 for _ in range(inp.ndim - 1)]).float()
        return lam * inp + (1 - lam) * inp[index, :]

    def _mix_target(
        self,
        lam: Tensor | float,
        target: Tensor,
        index: Tensor,
    ) -> Tensor:
        y1 = F.one_hot(target, self.num_classes)
        y2 = F.one_hot(target[index], self.num_classes)
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1 for _ in range(y1.ndim - 1)]).float()

        return lam * y1 + (1 - lam) * y2

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Apply mixup on a batch of inputs and targets.

        Args:
            x (Tensor): input or input batch.
            y (Tensor): target or target batch.

        Returns:
            tuple[Tensor, Tensor]: mixed-up inputs and targets.
        """


class Mixup(AbstractMixup):
    """Original Mixup method from Zhang et al.

    Reference:
        "mixup: Beyond Empirical Risk Minimization" (ICLR 2021)
        http://arxiv.org/abs/1710.09412.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        mixed_x = self._linear_mixing(lam, x, index)
        mixed_y = self._mix_target(lam, y, index)
        return mixed_x, mixed_y


class MixupMP(AbstractMixup):
    """Original Mixup method from Zhang et al. modified for MixupMP.

    In this implementation, we return both the mixuped and the normal
    inputs and targets, all concatenated.

    Reference:
        "mixup: Beyond Empirical Risk Minimization" (ICLR 2021)
        http://arxiv.org/abs/1710.09412.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        mixed_x = self._linear_mixing(lam, x, index)
        mixed_y = self._mix_target(lam, y, index)
        return torch.cat([mixed_x, x]), torch.cat([mixed_y, y])


class MixupIO(AbstractMixup):
    """Mixup on inputs only with targets unchanged, from Wang et al.

    Reference:
        "On the Pitfall of Mixup for Uncertainty Calibration" (CVPR 2023)
        https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        mixed_x = self._linear_mixing(lam, x, index)

        if self.isobatch:
            mixed_y = self._mix_target(float(lam > 0.5), y, index)
        else:
            mixed_y = self._mix_target((lam > 0.5).float(), y, index)
        return mixed_x, mixed_y


class RegMixup(AbstractMixup):
    """RegMixup method from Pinto et al.

    Reference:
        'RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness' (NeurIPS 2022)
        https://arxiv.org/abs/2206.14502.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        part_x = self._linear_mixing(lam, x, index)
        part_y = self._mix_target(lam, y, index)
        mixed_x = torch.cat([x, part_x], dim=0)
        mixed_y = torch.cat([F.one_hot(y, self.num_classes), part_y], dim=0)
        return mixed_x, mixed_y


class WarpingMixup(AbstractMixup):
    def __init__(
        self,
        alpha: float = 1.0,
        mode: str = "batch",
        num_classes: int = 1000,
        apply_kernel: bool = True,
        tau_max: float = 1.0,
        tau_std: float = 0.5,
    ) -> None:
        """Kernel Warping Mixup method from Bouniot et al.

        Reference:
            "Tailoring Mixup to Data using Kernel Warping functions" (2023)
            https://arxiv.org/abs/2311.01434.
        """
        if not scipy_installed:  # coverage: ignore
            raise ImportError(
                "The scipy library is not installed. Please install"
                "torch_uncertainty with the all option:"
                """pip install -U "torch_uncertainty[all]"."""
            )
        super().__init__(alpha=alpha, mode=mode, num_classes=num_classes)
        self.apply_kernel = apply_kernel
        self.tau_max = tau_max
        self.tau_std = tau_std

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        feats: Tensor,
        warp_param: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        if self.apply_kernel:
            l2_dist = (
                (feats - feats[index])
                .pow(2)
                .sum([i for i in range(len(feats.size())) if i > 0])
                .cpu()
                .numpy()
            )
            warp_param = sim_gauss_kernel(l2_dist, self.tau_max, self.tau_std)

        k_lam = torch.as_tensor(beta_warping(lam, warp_param), device=x.device)
        mixed_x = self._linear_mixing(k_lam, x, index)
        mixed_y = self._mix_target(k_lam, y, index)
        return mixed_x, mixed_y


# ruff: noqa: ARG001
def build_mixup(
    mixtype: str,
    mixup_alpha: float,
    cutmix_alpha: float | None,
    isobatch: bool,
    num_classes: int,
    kernel_tau_max: float | None = None,
    kernel_tau_std: float | None = None,
    **kwargs: Any,
):
    if mixup_alpha < 0 or (cutmix_alpha is not None and cutmix_alpha < 0):
        raise ValueError(
            f"Cutmix alpha and Mixup alpha must be positive. Got {mixup_alpha} and {cutmix_alpha}."
        )

    factories = {
        "timm": lambda: timm_Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mode="batch" if isobatch else "elem",
            num_classes=num_classes,
        ),
        "mixup": lambda: Mixup(
            alpha=mixup_alpha,
            isobatch=isobatch,
            num_classes=num_classes,
        ),
        "mixup_io": lambda: MixupIO(
            alpha=mixup_alpha,
            isobatch=isobatch,
            num_classes=num_classes,
        ),
        "regmixup": lambda: RegMixup(
            alpha=mixup_alpha,
            isobatch=isobatch,
            num_classes=num_classes,
        ),
        "kernel_warping": lambda: WarpingMixup(
            alpha=mixup_alpha,
            isobatch=isobatch,
            num_classes=num_classes,
            apply_kernel=True,
            tau_max=kernel_tau_max,
            tau_std=kernel_tau_std,
        ),
    }
    return factories.get(mixtype, Identity)()
