from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import Conformal


class ConformalClsTHR(Conformal):
    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        ts_init_val: float = 1.0,
        ts_lr: float = 0.1,
        ts_max_iter: int = 100,
        enable_ts: bool = True,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Threshold-based conformal classifier (THR; Sadinle et al., 2019).

        The simplest conformal classification rule. Defines the non-conformity score
        of class :math:`c` as :math:`s(\mathbf{x}, c) = 1 - \hat{p}_c(\mathbf{x})`,
        calibrates the empirical :math:`(1 - \alpha)`-quantile :math:`\hat{q}` of the
        scores at the true class on a held-out calibration set, and at test time
        outputs the prediction set

        .. math::
            \mathcal{C}(\mathbf{x}) = \{ c : \hat{p}_c(\mathbf{x}) \geq 1 - \hat{q} \},

        guaranteeing marginal coverage of :math:`1 - \alpha`. The top-1 class is
        always included to avoid empty sets. Probabilities can optionally be calibrated
        first via temperature scaling (``enable_ts=True``), which usually yields smaller
        prediction sets.

        Args:
            alpha: Target mis-coverage level :math:`\alpha \in (0, 1)`.
            model: Model to be calibrated. Defaults to ``None``.
            ts_init_val: Initial value for the temperature. Defaults to ``1.0``.
            ts_lr: Learning rate for the temperature scaling optimizer. Defaults to ``0.1``.
            ts_max_iter: Maximum number of iterations for the temperature scaling
                optimizer. Defaults to ``100``.
            enable_ts: Whether to scale the logits via temperature scaling before
                computing the conformal scores. Defaults to ``True``.
            device: Device to use. Defaults to ``None``.

        Warning:
            This implementation only works in the multiclass setting. Raise an issue
            if binary support is needed.

        Reference:
            - `Sadinle, M., Lei, J., & Wasserman, L. (2019). Least Ambiguous Set-Valued
              Classifiers with Bounded Error Levels <https://arxiv.org/abs/1609.00451>`_.

        Code inspired by TorchCP.
        """
        super().__init__(
            alpha=alpha,
            model=model,
            ts_init_val=ts_init_val,
            ts_lr=ts_lr,
            ts_max_iter=ts_max_iter,
            enable_ts=enable_ts,
            device=device,
        )

    def fit(self, dataloader: DataLoader) -> None:
        assert self.model is not None
        if self.enable_ts:
            self.model.fit(dataloader=dataloader)

        logit_list = []
        label_list = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logit_list.append(self.model_forward(inputs))
                label_list.append(labels)

        probs = torch.cat(logit_list)
        labels = torch.cat(label_list).long()
        true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        scores = 1.0 - true_class_probs
        self.q_hat = torch.quantile(scores, 1.0 - self.alpha)

    @torch.no_grad()
    def conformal(self, inputs: Tensor) -> Tensor:
        """Perform conformal prediction on the test set."""
        probs = self.model_forward(inputs)
        pred_set = probs >= 1.0 - self.quantile
        top1 = torch.argmax(probs, dim=1, keepdim=True)
        pred_set.scatter_(1, top1, True)  # Always include top-1 class
        return pred_set.float() / pred_set.sum(dim=1, keepdim=True)
