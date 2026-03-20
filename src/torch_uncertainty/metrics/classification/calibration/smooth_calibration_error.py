import logging
from typing import Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .smooth_calibration_kernels import LogitGaussianKernel, ReflectedGaussianKernel


class SmoothCalibrationError(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    confidences: list[Tensor]
    accuracies: list[Tensor]
    final_bandwidth: float | None = None

    def __init__(
        self,
        kernel_type: Literal["logit", "reflected"] = "logit",
        bandwidth: float | Literal["auto"] = "auto",
        eps: float = 0.001,
        mesh_pts: int = 200,
        refine_steps: int = 10,
        **kwargs,
    ):
        """Smooth Expected Calibration Error (SmECE).

        This metric implements the Kernel Density Estimation based ECE as
        proposed by Błasiok & Nakkiran (2023). It addresses the limitations of
        standard binned ECE, such as bin-edge effects and poor resolution for
        overconfident models, by using a continuous kernel and an adaptive
        bandwidth selection strategy. Computed on the top label.

        Args:
            kernel_type (str, optional): The kernel to use. Choose between:
                - ``'logit'``: Applies a Gaussian kernel in log-odds space. This
                    effectively uses an adaptive bandwidth that is narrower near 1.0,
                    making it ideal for modern overconfident models. (Default)
                - ``'reflected'``: Applies a Gaussian kernel in probability space
                    with reflections at 0 and 1 to prevent boundary bias.
                Note that relplot's original implementation has ``'reflected'`` as default.
            bandwidth (Literal[auto] | float, optional): The kernel bandwidth $h$. If set to
                ``'auto'``, it uses a fixed-point binary search to find a bandwidth
                consistent with the error level. Defaults to ``'auto'``.
            eps (float, optional): The tolerance for the binary search when
                bandwidth is ``'auto'``. Defaults to ``0.001``.
            mesh_pts (int, optional): The base number of points for the grid
                discretization. The actual number may be higher depending on the
                bandwidth. Defaults to ``200``.
            refine_steps (int, optional): Number of binary search iterations for
                the ``'auto'`` bandwidth. Defaults to ``10``.
            **kwargs: Additional arguments for the :class:`torchmetrics.Metric` base.

        Note:
            In the multiclass case, this metric evaluates the calibration of the
            maximum probability (top-label calibration). In the binary case, it
            evaluates the calibration of the predicted class (i.e., using $max(p, 1-p)$).

        Note:
            This implementation has been tested on a use case and provided the same values
            (with 6 equal significant figures) as relplot's original implementation.

        References:
            - Błasiok, J. & Nakkiran, P. Smooth ECE: Principled Reliability
            Diagrams. ICLR 2024.
        """
        super().__init__(**kwargs)
        if kernel_type not in ["logit", "reflected"]:
            raise ValueError(f"kernel_type must be 'logit' or 'reflected'. Got {kernel_type}.")
        if not isinstance(bandwidth, float) and bandwidth != "auto":
            raise ValueError(f"Invalid bandwidth: {bandwidth}.")

        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.eps = eps
        self.mesh_pts = mesh_pts
        self.refine_steps = refine_steps

        self.add_state("confidences", default=[], dist_reduce_fx="cat")
        self.add_state("accuracies", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with predictions and targets.

        Args:
            preds (Tensor): Predictions from the model.
                - Multiclass: Shape ``(N, C)`` (logits or probabilities).
                - Binary: Shape ``(N,)`` or ``(N, 1)`` (logits or probabilities).
            target (Tensor): Ground truth labels.
                - Multiclass: Shape ``(N,)`` containing class indices.
                - Binary: Shape ``(N,)`` containing 0 or 1.
        """
        if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
            preds = preds.view(-1)
            target = target.view(-1)

            if preds.max() > 1.0 or preds.min() < 0.0:  # coverage: ignore
                logging.warning("Smooth ECE: the inputs are not probabilities, applying sigmoid.")
                probs = torch.sigmoid(preds)
            else:
                probs = preds

            conf = torch.where(probs >= 0.5, probs, 1.0 - probs)
            pred_labels = (probs >= 0.5).long()
            acc = (pred_labels == target).float()
        else:
            if preds.max() > 1.0 or preds.min() < 0.0:  # coverage: ignore
                logging.warning("Smooth ECE: the inputs are not probabilities, applying softmax.")
                preds = torch.softmax(preds, dim=-1)
            conf, pred_labels = torch.max(preds, dim=-1)
            acc = (pred_labels == target).view(-1).float()

        self.confidences.append(conf)
        self.accuracies.append(acc)

    def _compute_smooth_ece(self, conf: Tensor, acc: Tensor, bandwidth: float) -> Tensor:
        ev_pts = max(int(10 / bandwidth), self.mesh_pts)
        t = torch.linspace(0, 1, ev_pts, device=conf.device)

        # Initialize the appropriate kernel
        if self.kernel_type == "logit":
            kernel = LogitGaussianKernel(bandwidth)
        else:
            kernel = ReflectedGaussianKernel(bandwidth)

        residuals = conf - acc
        ys, dens = kernel.smooth(conf, residuals, t)
        ys, dens = ys.float(), dens.float()

        valid_mask = dens > 1e-8
        rs = torch.zeros_like(ys)
        rs[valid_mask] = torch.abs(ys[valid_mask])
        return torch.sum(rs * dens) / (dens.sum() + 1e-8)

    def _search_bandwidth(self, conf: Tensor, acc: Tensor) -> float:
        def check_smooth_ece(alpha: float) -> bool:
            if alpha < self.eps:  # coverage: ignore
                return True
            return alpha < self.eps or alpha < self._compute_smooth_ece(conf, acc, alpha).item()

        start, end = 1.0, 0.0
        if check_smooth_ece(start):  # coverage: ignore
            return start

        for _ in range(self.refine_steps):
            midpoint = (start + end) / 2.0
            if check_smooth_ece(midpoint):
                end = midpoint
            else:
                start = midpoint
        return start

    def compute(self) -> Tensor:
        """Compute the Smooth ECE based on the accumulated state.

        Returns:
            Tensor: The scalar SmECE value.
        """
        conf = dim_zero_cat(self.confidences)
        acc = dim_zero_cat(self.accuracies)

        if isinstance(self.bandwidth, float):
            self.final_bandwidth = self.bandwidth
        else:  # if self.bandwidth == "auto":
            self.final_bandwidth = self._search_bandwidth(conf, acc)
        logging.info("Selected bandwidth: %s", self.final_bandwidth)
        return self._compute_smooth_ece(conf, acc, self.final_bandwidth)
