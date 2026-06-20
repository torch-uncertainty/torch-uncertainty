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
        r"""The Deep Evidential Classification (DEC) loss.

        Trains a classifier to output Dirichlet evidence :math:`\mathbf{e} \in
        \mathbb{R}_{\geq 0}^C` instead of class probabilities. The Dirichlet parameters
        are :math:`\boldsymbol{\alpha} = \mathbf{e} + 1` and the total evidence
        :math:`S = \sum_c \alpha_c` controls the predictive uncertainty (smaller
        :math:`S` ⇒ more uncertainty). The full loss is the sum of an expected
        cross-entropy term (selected by :attr:`loss_type`) and a KL regulariser that
        pushes evidence on incorrect classes towards zero, annealed by either a
        constant :attr:`reg_weight` or a linear schedule of length
        :attr:`annealing_step`.

        Args:
            annealing_step: Annealing step for the weight of the
                regularization term. Defaults to ``None``.
            reg_weight: Fixed weight of the regularization term. Defaults to ``None``.
            loss_type: Specifies the loss type to apply to the
                Dirichlet parameters: ``'mse'`` | ``'log'`` | ``'digamma'``.
            reduction: Specifies the reduction to apply to the
                output: ``'none'`` | ``'mean'`` | ``'sum'``.

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
            annealing_coef = torch.tensor(0.0, dtype=evidence.dtype, device=evidence.device)
        elif self.annealing_step is None and self.reg_weight is not None:
            annealing_coef = torch.tensor(
                float(self.reg_weight),
                dtype=evidence.dtype,
                device=evidence.device,
            )
        else:
            if current_epoch is None:  # coverage: ignore
                raise ValueError(
                    "current_epoch must be set when annealing_step is used and reg_weight is None."
                )
            if self.annealing_step is None:  # coverage: ignore
                raise ValueError(
                    "annealing_step must be set when annealing_step is used and reg_weight is None."
                )
            annealing_coef = torch.min(
                input=torch.tensor(1.0, dtype=evidence.dtype, device=evidence.device),
                other=torch.tensor(
                    float(current_epoch) / float(self.annealing_step),
                    dtype=evidence.dtype,
                    device=evidence.device,
                ),
            )

        loss = loss_dirichlet + annealing_coef * self._kldiv_reg(evidence, targets)
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
        r"""The Confidence Penalty loss.

        Augments the standard cross-entropy loss with a regulariser that penalises
        low-entropy (over-confident) predictive distributions:

        .. math::
            \mathcal{L} = \text{CE}(\mathbf{z}, y)
            + \lambda \left( \log C + \sum_{i=1}^{C} p_i \log(p_i + \varepsilon) \right),

        where :math:`\mathbf{p} = \mathrm{softmax}(\mathbf{z})`, :math:`C` is the number
        of classes, :math:`\lambda` is :attr:`reg_weight`, and :math:`\varepsilon` is
        :attr:`eps`. The regulariser equals the negative entropy of :math:`\mathbf{p}`
        shifted by :math:`\log C` so that it is non-negative.

        Args:
            reg_weight: The weight :math:`\lambda` of the regularization term.
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
            eps: A small value to avoid numerical instability.
                Defaults to ``1e-6``.

        References:
            [1] `Pereyra, G., et al. (2017). Regularizing neural networks by penalizing
            confident output distributions <https://arxiv.org/pdf/1701.06548>`_.
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
            logits: The inputs of the Bayesian Neural Network
            targets: The target values

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

        Combines the standard cross-entropy with a *conflictual* regulariser that
        encourages the model to assign non-negligible probability to a uniformly random
        class :math:`c^\star`:

        .. math::
            \mathcal{L} = \text{CE}(\mathbf{z}, y)
            - \lambda \log p_{c^\star}, \quad c^\star \sim \mathrm{Uniform}(1, C).

        This counteracts the natural tendency of cross-entropy to collapse all
        probability mass on a single class and improves the calibration of epistemic
        uncertainty estimates.

        Args:
            reg_weight: The weight :math:`\lambda` of the regularization term.
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``.

        References:
            [1] `Fellaji, M., et al. (2024). On the Calibration of Epistemic Uncertainty:
            Principles, Paradoxes and Conflictual Loss
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
            logits: The outputs of the model.
            targets: The target values.

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
        r"""Focal Loss for classification tasks.

        Down-weights the contribution of well-classified samples to the cross-entropy
        loss, focusing training on hard examples. For a target class :math:`y` with
        predicted probability :math:`p_y`,

        .. math::
            \text{FL}(p_y) = -\alpha_y \, (1 - p_y)^\gamma \, \log p_y,

        where :math:`\gamma \geq 0` is :attr:`gamma` (a larger :math:`\gamma`
        suppresses easy examples more aggressively) and :math:`\alpha_y` is the
        per-class weight from :attr:`alpha` (defaulting to ``1``).

        Args:
            gamma: The focusing parameter :math:`\gamma`, as described in the paper.
            alpha: Per-class rescaling weights. Defaults to ``None``.
            reduction: ``'mean'``, ``'sum'`` or ``'none'``. Defaults to ``'mean'``.

        References:
            [1] `Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection
            <https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf>`_.

            [2] Inspired by https://github.com/AdeelH/pytorch-multi-class-focal-loss.
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
            weight: A manual rescaling weight given to the
                loss of each batch element. If given, has to be a Tensor of size
                "nbatch". Defaults to ``None``.
            reduction: Specifies the reduction to apply to the
                output: ``'none'`` | ``'mean'`` | ``'sum``'. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Defaults
                to ``'mean'``.
            label_smoothing: The label smoothing factor. Defaults
                to ``0.0``.
        """
        super().__init__(weight=weight, reduction=reduction)
        if label_smoothing < 0:
            raise ValueError(
                "The label smoothing term of the BCE loss should be non-negative, but got "
                f"{label_smoothing}."
            )
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: A002
        if self.label_smoothing == 0.0:
            return super().forward(input, target.type_as(input))
        target = target.float()
        target = target * (1 - self.label_smoothing) + self.label_smoothing / 2
        loss = target * F.logsigmoid(input) + (1 - target) * F.logsigmoid(-input)
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
        reduction = "none" if reduction is None else reduction
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.max_sup = max_sup

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: A002
        if self.max_sup == 0.0:
            return super().forward(input, target)
        z_top1 = input.topk(1, -1)[0]
        reg = z_top1 - input.mean(-1, keepdim=True)
        loss = (
            F.cross_entropy(
                input,
                target,
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
        consists of **mixup-augmented samples** followed by **original samples**.
        The `mixup_ratio` (r) controls the number of mixup samples relative to
        normal samples produced by the transform:

          - r <= 1.0: selects fewer mixup samples,
          - r > 1.0: selects more mixup samples.

        Both branches use cross-entropy (or KL divergence for soft targets)
        weighted according to the ratio r.

        Args:
            mixup_ratio: Ratio of number of mixup samples vs normal samples
                output by the MixupMP transform. This should match the transform's
                :attr:`mixup_ratio` hyperparameter. Defaults to ``1.0`` (equal
                weighting).
            weight: a manual rescaling weight given to each class.
            ignore_index: Specifies a target value that is ignored and does not
                contribute to the input gradient. Only applies to class-index
                (long) targets. Defaults to ``-100``.
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``.

        Raises:
            ValueError: if ``mixup_ratio`` is not strictly positive.

        See Also:
            torch_uncertainty/transforms/mixup.py — MixupMP transform implementation.

        Reference:
            "Posterior Uncertainty Quantification in Neural Networks using Data
            Augmentation" (AISTATS 2024) by Luhuan Wu & Sinead Williamson.
        """
        if mixup_ratio <= 0:
            raise ValueError(f"mixup_ratio must be > 0. Got {mixup_ratio}.")
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.mixup_ratio = mixup_ratio

    def _branch_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Per-branch loss with empty-slice short-circuit and soft-label dispatch.

        An empty slice contributes a 0-D zero so that whichever branch is empty
        (depending on ``mixup_ratio``) does not invoke the underlying
        cross-entropy. torch >= 2.12 validates target dtype even on zero-row
        inputs, which would otherwise raise ``RuntimeError`` for soft-label
        batches that fall entirely on the opposite branch.

        Soft (float) targets use KL divergence to match the paper's formulation;
        class-index (long) targets use the parent class' cross-entropy.
        """
        if preds.size(0) == 0:
            return preds.new_zeros(())
        if targets.dtype.is_floating_point:
            log_prob = F.log_softmax(preds, dim=-1)
            return F.kl_div(
                log_prob,
                targets,
                reduction="batchmean" if self.reduction == "mean" else self.reduction,
            )
        return super().forward(preds, targets)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: A002
        """Compute the MixupMP loss.

        The MixupMP transform concatenates ``mixup_count`` mixup-augmented
        samples followed by the remaining original samples. This loss splits
        the batch at that boundary and weights the mixup branch by
        ``mixup_ratio``.

        Args:
            input: model logits of shape ``(N_total, num_classes)``.
            target: target labels — either class indices (``long``, shape
                ``(N_total,)``) or soft labels (``float``, shape
                ``(N_total, num_classes)``).
        """
        mixup_count = round((self.mixup_ratio / (self.mixup_ratio + 1)) * input.size(0))

        loss_mixup = self._branch_loss(input[:mixup_count], target[:mixup_count])
        loss_norm = self._branch_loss(input[mixup_count:], target[mixup_count:])

        # unnormalized as in the paper's implementation
        return self.mixup_ratio * loss_mixup + loss_norm
