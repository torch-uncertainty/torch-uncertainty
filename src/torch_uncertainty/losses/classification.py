import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DECLoss(nn.Module):
    def __init__(
        self,
        annealing_step: int | None = None,
        reg_weight: float | None = None,
        loss_type: str = "log",
        reduction: str | None = "mean",
    ) -> None:
        """The deep evidential classification loss.

        Args:
            annealing_step (int): Annealing step for the weight of the
                regularization term.
            reg_weight (float): Fixed weight of the regularization term.
            loss_type (str, optional): Specifies the loss type to apply to the
                Dirichlet parameters: ``'mse'`` | ``'log'`` | ``'digamma'``.
            reduction (str, optional): Specifies the reduction to apply to the
                output:``'none'`` | ``'mean'`` | ``'sum'``.

        References:
            [1] `Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. NeurIPS 2018
            <https://arxiv.org/abs/1806.01768>`_.

        """
        super().__init__()

        if reg_weight is not None and (reg_weight < 0):
            raise ValueError(
                f"The regularization weight should be non-negative, but got {reg_weight}."
            )
        self.reg_weight = reg_weight

        if annealing_step is not None and (annealing_step <= 0):
            raise ValueError(f"The annealing step should be positive, but got {annealing_step}.")
        self.annealing_step = annealing_step

        if reduction not in ("none", "mean", "sum") and reduction is not None:
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

        if loss_type not in ["mse", "log", "digamma"]:
            raise ValueError(f"{loss_type} is not a valid value for mse/log/digamma loss.")
        self.loss_type = loss_type

    def _mse_loss(self, evidence: Tensor, targets: Tensor) -> Tensor:
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0
        strength = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((targets - (alpha / strength)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (strength - alpha) / (strength * strength * (strength + 1)),
            dim=1,
            keepdim=True,
        )
        return loglikelihood_err + loglikelihood_var

    def _log_loss(self, evidence: Tensor, targets: Tensor) -> Tensor:
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return torch.sum(
            targets * (torch.log(strength) - torch.log(alpha)),
            dim=1,
            keepdim=True,
        )

    def _digamma_loss(self, evidence: Tensor, targets: Tensor) -> Tensor:
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return torch.sum(
            targets * (torch.digamma(strength) - torch.digamma(alpha)),
            dim=1,
            keepdim=True,
        )

    def _kldiv_reg(
        self,
        evidence: Tensor,
        targets: Tensor,
    ) -> Tensor:
        num_classes = evidence.size()[-1]
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0

        kl_alpha = (alpha - 1) * (1 - targets) + 1

        ones = torch.ones([1, num_classes], dtype=evidence.dtype, device=evidence.device)
        sum_kl_alpha = torch.sum(kl_alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_kl_alpha)
            - torch.lgamma(kl_alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = torch.sum(
            (kl_alpha - ones) * (torch.digamma(kl_alpha) - torch.digamma(sum_kl_alpha)),
            dim=1,
            keepdim=True,
        )
        return first_term + second_term

    def forward(
        self,
        evidence: Tensor,
        targets: Tensor,
        current_epoch: int | None = None,
    ) -> Tensor:
        if self.annealing_step is not None and self.annealing_step > 0 and current_epoch is None:
            raise ValueError(
                "The epoch num should be positive when \
                annealing_step is settled, but got "
                f"{current_epoch}."
            )

        if targets.ndim != 1:  # if no mixup or cutmix
            raise NotImplementedError("DECLoss does not yet support mixup/cutmix.")
        # TODO: handle binary
        targets = F.one_hot(targets, num_classes=evidence.size()[-1])

        if self.loss_type == "mse":
            loss_dirichlet = self._mse_loss(evidence, targets)
        elif self.loss_type == "log":
            loss_dirichlet = self._log_loss(evidence, targets)
        else:  # self.loss_type == "digamma"
            loss_dirichlet = self._digamma_loss(evidence, targets)

        if self.reg_weight is None and self.annealing_step is None:
            annealing_coef = 0
        elif self.annealing_step is None and self.reg_weight > 0:
            annealing_coef = self.reg_weight
        else:
            annealing_coef = torch.min(
                input=torch.tensor(1.0, dtype=evidence.dtype),
                other=torch.tensor(current_epoch / self.annealing_step, dtype=evidence.dtype),
            )

        loss_reg = self._kldiv_reg(evidence, targets)
        loss = loss_dirichlet + annealing_coef * loss_reg
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class ConfidencePenaltyLoss(nn.Module):
    def __init__(
        self,
        reg_weight: float = 1,
        reduction: str | None = "mean",
        eps: float = 1e-6,
    ) -> None:
        """The Confidence Penalty Loss.

        Args:
            reg_weight (float, optional): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
                output:``'none'`` | ``'mean'`` | ``'sum'``. Defaults to "mean".
            eps (float, optional): A small value to avoid numerical instability.
                Defaults to ``1e-6.``

        References:
            [1] `Gabriel Pereyra: Regularizing neural networks by penalizing confident output distributions
            <https://arxiv.org/pdf/1701.06548>`_.

        """
        super().__init__()
        if reduction is None:
            reduction = "none"
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

        if eps < 0:
            raise ValueError(f"The epsilon value should be non-negative, but got {eps}.")
        self.eps = eps
        if reg_weight < 0:
            raise ValueError(
                f"The regularization weight should be non-negative, but got {reg_weight}."
            )
        self.reg_weight = reg_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the Confidence Penalty loss.

        Args:
            logits (Tensor): The inputs of the Bayesian Neural Network
            targets (Tensor): The target values

        Returns:
            Tensor: The Confidence Penalty loss
        """
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        reg_loss = torch.log(torch.tensor(logits.shape[-1], device=probs.device)) + (
            probs * torch.log(probs + self.eps)
        ).sum(dim=-1)
        if self.reduction == "sum":
            return ce_loss + self.reg_weight * reg_loss.sum()
        if self.reduction == "mean":
            return ce_loss + self.reg_weight * reg_loss.mean()
        return ce_loss + self.reg_weight * reg_loss


class ConflictualLoss(nn.Module):
    def __init__(
        self,
        reg_weight: float = 1,
        reduction: str | None = "mean",
    ) -> None:
        r"""The Conflictual Loss.

        Args:
            reg_weight (float, optional): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
                output:``'none'`` | ``'mean'`` | ``'sum'``.

        References:
            [1] `Mohammed Fellaji et al. On the Calibration of Epistemic Uncertainty: Principles, Paradoxes and Conflictual Loss
            <https://arxiv.org/pdf/2407.12211>`_.

        """
        super().__init__()
        if reduction is None:
            reduction = "none"
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction
        if reg_weight < 0:
            raise ValueError(
                f"The regularization weight should be non-negative, but got {reg_weight}."
            )
        self.reg_weight = reg_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the conflictual loss.

        Args:
            logits (Tensor): The outputs of the model.
            targets (Tensor): The target values.

        Returns:
            Tensor: The conflictual loss.
        """
        class_index = torch.randint(
            0, logits.shape[-1], (1,), dtype=torch.long, device=logits.device
        )
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        reg_loss = -F.log_softmax(logits, dim=1)[:, class_index]
        if self.reduction == "sum":
            return ce_loss + self.reg_weight * reg_loss.sum()
        if self.reduction == "mean":
            return ce_loss + self.reg_weight * reg_loss.mean()
        return ce_loss + self.reg_weight * reg_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float,
        alpha: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        """Focal-Loss for classification tasks.

        Args:
            gamma (float, optional): A constant, as described in the paper.
            alpha (Tensor, optional): Weights for each class. Defaults to ``None``.
            reduction (str, optional): ``'mean'``, ``'sum'`` or ``'none'``. Defaults to ``'mean'``.

        References:
            [1] `Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection.
            <https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf>`_.

            [2] Inspired by https://github.com/AdeelH/pytorch-multi-class-focal-loss .

        """
        if reduction not in ("none", "mean", "sum") and reduction is not None:
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

        if gamma < 0:
            raise ValueError(
                f"The gamma term of the focal loss should be non-negative, but got {gamma}."
            )
        self.gamma = gamma

        super().__init__()
        self.alpha = alpha
        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        log_p = F.log_softmax(inputs, dim=-1)
        ce = self.nll_loss(log_p, targets)

        all_rows = torch.arange(len(inputs))
        log_pt = log_p[all_rows, targets]

        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEWithLogitsLSLoss(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        weight: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        """Binary Cross Entropy with Logits Loss with label smoothing.

        The original PyTorch implementation of the BCEWithLogitsLoss does not
        support label smoothing. This implementation adds label smoothing to
        the BCEWithLogitsLoss.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to the
                loss of each batch element. If given, has to be a Tensor of size
                "nbatch". Defaults to ``None``.
            reduction (str, optional): Specifies the reduction to apply to the
                output: ``'none'`` | ``'mean'`` | ``'sum``'. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Defaults
                to ``'mean'``.
            label_smoothing (float, optional): The label smoothing factor. Defaults
                to ``0.0``.
        """
        super().__init__(weight=weight, reduction=reduction)
        if label_smoothing < 0:
            raise ValueError(
                "The label smoothing term of the BCE loss should be non-negative, but got "
                f"{label_smoothing}."
            )
        self.label_smoothing = label_smoothing

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.label_smoothing == 0.0:
            return super().forward(inputs, targets.type_as(inputs))
        targets = targets.float()
        targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2
        loss = targets * F.logsigmoid(inputs) + (1 - targets) * F.logsigmoid(-inputs)
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduction == "mean":
            return -loss.mean()
        if self.reduction == "sum":
            return -loss.sum()
        return -loss


class CrossEntropyMaxSupLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        label_smoothing: float = 0,
        max_sup: float = 0,
        *,
        weight: Tensor | None = None,
        size_average=None,
        reduction: str | None = "mean",
    ) -> None:
        """Max suppression cross-entropy loss.

        Note: We haven't implemented the linear loss scheduler suggested in the paper. Raise
            an issue if needed.

        Reference:
            MaxSup: Fixing Label-smoothing for improved feature representation. Y. Zhou et al.
        """
        super().__init__(weight, size_average, reduction=reduction, label_smoothing=label_smoothing)
        self.max_sup = max_sup

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.max_sup == 0.0:
            return super().forward(inputs, targets)
        z_top1 = inputs.topk(1, -1)[0]
        reg = z_top1 - inputs.mean(-1, keepdim=True)
        loss = (
            F.cross_entropy(
                inputs,
                targets,
                label_smoothing=self.label_smoothing,
            )
            + self.max_sup * reg
        )
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MixupMPLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        mixup_ratio: float = 1.0,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        """MixupMP loss from Wu & Williamson.

        When using the MixupMP transform, the batch returned to the model
        consists of **mixup-augmented samples** and **original samples** concatenated.
        The `mixup_ratio` (r) controls the number of mixup samples relative to normal samples
        produced by the transform:

          - r <= 1.0: selects fewer mixup samples,
          - r > 1.0: selects more mixup samples.

        Both mixup and normal samples use cross-entropy but should be **weighted** according to the
        ratio r.

        Args:
            mixup_ratio (float): Ratio of number of mixup samples vs normal samples
                output by the MixupMP transform. This should match the transform's
                :attr:`mixup_ratio` hyperparameter. Defaults to ``1.0`` (equal weighting).
            weight (Tensor | None): a manual rescaling weight given to each class.
            ignore_index (int): Specifies a target value that is ignored
                and does not contribute to the input gradient. Defaults to ``-100``.
            reduction (str): Specifies the reduction to apply to the output: 'none'|'mean'|'sum'.

        See Also:
            torch_uncertainty/transforms/mixup.py — MixupMP transform implementation.

        Reference:
            "Posterior Uncertainty Quantification in Neural Networks using Data Augmentation"
            (AISTATS 2024) by Luhuan Wu & Sinead Williamson. :contentReference[oaicite:1]{index=1}
        """
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        if mixup_ratio <= 0:
            raise ValueError(f"mixup_ratio must be > 0. Got {mixup_ratio}.")
        self.mixup_ratio = mixup_ratio

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """The mixup transform should arrange outputs as `[mixup, normal]` or
        `[normal, mixup]` depending on r; this splits them accordingly.

        Mixup targets may be soft labels (one-hot float tensor), so we handle
        that case by manually using F.kl_div if needed.

        Args:
            inputs (Tensor): model logits shape (N_total, num_classes)
            targets (Tensor): target labels (one-hot or class indices) shape (N_total, ...)
        """
        # compute expected counts based on mixup_ratio
        r = self.mixup_ratio

        num_total_samples = inputs.size(0)
        # determine how many samples correspond to mixup vs normal
        if r <= 1.0:
            mixup_count = round((r / (1.0 + r)) * num_total_samples)
            normal_count = num_total_samples - mixup_count
        else:
            normal_count = round(((1.0 / r) / (1.0 + (1.0 / r))) * num_total_samples)
            mixup_count = num_total_samples - normal_count

        # slices: assume mixup first, then normal
        mix_slice = slice(0, mixup_count)
        norm_slice = slice(mixup_count, mixup_count + normal_count)

        mixup_preds = inputs[mix_slice]
        mixup_targets = targets[mix_slice]
        norm_preds = inputs[norm_slice]
        norm_targets = targets[norm_slice]

        # standard cross entropy for normal samples
        loss_norm = super().forward(norm_preds, norm_targets)

        # mixup may have soft labels
        if mixup_targets.dtype.is_floating_point:
            # use KL divergence for soft labels
            log_prob = F.log_softmax(mixup_preds, dim=-1)
            loss_mixup = F.kl_div(
                log_prob,
                mixup_targets,
                reduction="batchmean" if self.reduction == "mean" else self.reduction,
            )
        else:
            loss_mixup = super().forward(mixup_preds, mixup_targets)

        # unnormalized as in the implementation
        return r * loss_mixup + loss_norm
