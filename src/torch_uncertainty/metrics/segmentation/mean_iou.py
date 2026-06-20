from torch import Tensor
from torchmetrics.classification.stat_scores import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide


class MeanIntersectionOverUnion(MulticlassStatScores):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        r"""Computes the Mean Intersection over Union (mIoU) score.

        For a multi-class segmentation task with :math:`C` classes, the per-class
        Intersection over Union is

        .. math::
            \text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c},

        where :math:`\text{TP}_c`, :math:`\text{FP}_c`, :math:`\text{FN}_c` are the
        numbers of true-positive, false-positive and false-negative pixels for class
        :math:`c` (aggregated over all images). The mean IoU is the unweighted average

        .. math::
            \text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}_c.

        Classes that never appear in the targets are excluded from the average
        (``nanmean``).

        Args:
            num_classes: Integer specifying the number of classes.
            top_k: Number of highest probability or logit score predictions
                considered to find the correct label. Only works when ``preds`` contain
                probabilities/logits. Defaults to ``1``.
            ignore_index: Specifies a target value that is ignored and does
                not contribute to the metric calculation. Defaults to ``None``.
            validate_args: Bool indicating if input arguments and tensors should
                be validated for correctness. Set to ``False`` for faster computations. Defaults to
                ``True``.
            **kwargs: kwargs: Additional keyword arguments, see
                `Advanced metric settings <https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs>`_
                for more info.

        Shape:
            As input to ``forward`` and ``update`` the metric accepts the following input:

            - **preds** (:class:`~torch.Tensor`): An int tensor of shape ``(B, *)`` or float tensor of shape ``(B, C, *)``.
              If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
              probabilities/logits into an int tensor.
            - **target** (:class:`~torch.Tensor`): An int tensor of shape ``(B, *)``.

            As output to ``forward`` and ``compute`` the metric returns the following output:

            - **mean_iou** (:class:`~torch.Tensor`): The computed Mean Intersection over Union (IoU) score.
              A tensor containing a single float value.
        """
        super().__init__(
            num_classes,
            top_k,
            "macro",
            "global",
            ignore_index,
            validate_args,
            **kwargs,
        )

    def compute(self) -> Tensor:
        """Compute the Mean Intersection over Union (mIoU) based on the accumulated state."""
        tp, fp, _, fn = self._final_state()

        return _safe_divide(tp, tp + fp + fn, zero_division=float("nan")).nanmean()
