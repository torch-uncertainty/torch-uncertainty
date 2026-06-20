from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor, nn

from torch_uncertainty.metrics import MutualInformation, VariationRatio


class OODCriterionInputType(Enum):
    """Enum representing the type of input expected by the OOD (Out-of-Distribution) criteria.

    Attributes:
        LOGIT: The input of the OOD criterion is in the form of logits (pre-softmax values).
        PROB: The input is in the form of probabilities (post-softmax values).
        ESTIMATOR_PROB: The input is in the form of estimated probabilities from an ensemble
            or another probabilistic model.
        POST_PROCESSING: The input is the prediction score given by the post-processing method.
    """

    LOGIT = 1
    PROB = 2
    ESTIMATOR_PROB = 3
    POST_PROCESSING = 4


class TUOODCriterion(nn.Module, ABC):
    input_type: OODCriterionInputType
    single_only = False
    ensemble_only = False

    def __init__(self) -> None:
        """Abstract base class for Out-of-Distribution (OOD) criteria.

        This class defines a common interface for implementing various OOD detection
        criteria. Subclasses must implement the `forward` method.

        Attributes:
            input_type: Type of input expected by the criterion.
            ensemble_only: Whether the criterion requires ensemble outputs.
        """
        super().__init__()

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass for the OOD criterion.

        Args:
            inputs: The input tensor representing model outputs.

        Returns:
            Tensor: OOD score computed according to the criterion.
        """


class MaxLogitCriterion(TUOODCriterion):
    single_only = True
    input_type = OODCriterionInputType.LOGIT

    def __init__(self) -> None:
        r"""OOD criterion based on the Max-Logit score (Hendrycks et al.).

        Defined as the negative of the highest logit value (averaged over estimators
        in the ensemble case):

        .. math::
            \text{score}(\mathbf{z}) = -\max_{i} z_i.

        Lower maximum logits indicate greater uncertainty.

        Attributes:
            input_type: Expected input type is logits.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative of the maximum logit value.

        Args:
            inputs: Tensor of logits with shape ``(batch_size, num_estimators, num_classes)``.

        Returns:
            Tensor: Negative of the maximum logit value for each sample.
        """
        return -inputs.mean(dim=1).max(dim=-1).values


class EnergyCriterion(TUOODCriterion):
    single_only = True
    input_type = OODCriterionInputType.LOGIT

    def __init__(self) -> None:
        r"""OOD criterion based on the free-energy score (Liu et al., NeurIPS 2020).

        Defined as the negative log-sum-exp of the logits:

        .. math::
            \text{score}(\mathbf{z}) = -\log\left(\sum_{i=1}^{C} \exp(z_i)\right)

        where :math:`\mathbf{z} = [z_1, \dots, z_C]` is the logit vector. Larger values
        of the *energy* (i.e. of the score) indicate greater uncertainty and a higher
        likelihood of being out-of-distribution.

        Attributes:
            input_type: Expected input type is logits.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative energy score.

        Args:
            inputs: Tensor of logits with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative energy score for each sample.
        """
        return -inputs.mean(dim=1).logsumexp(dim=-1)


class MaxSoftmaxCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.PROB

    def __init__(self) -> None:
        r"""OOD criterion based on the Maximum Softmax Probability (MSP) baseline of
        Hendrycks & Gimpel (ICLR 2017).

        Defined as the negative of the maximum predicted class probability:

        .. math::
            \text{score}(\mathbf{p}) = -\max_{i} p_i,

        where :math:`\mathbf{p} = [p_1, \dots, p_C]` is the predictive distribution.
        Lower maximum probabilities indicate greater uncertainty.

        Attributes:
            input_type: Expected input type is probabilities.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative of the maximum softmax probability.

        Args:
            inputs: Tensor of probabilities with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative of the highest softmax probability for each sample.
        """
        return -inputs.max(-1)[0]


class PostProcessingCriterion(MaxSoftmaxCriterion):
    input_type = OODCriterionInputType.POST_PROCESSING


class DEUPCriterion(TUOODCriterion):
    """OOD criterion from DEUP epistemic uncertainty scores.

    Higher values indicate greater epistemic uncertainty (likely OOD or unreliable).
    Use with :class:`~torch_uncertainty.post_processing.DEUP`.
    """

    input_type = OODCriterionInputType.POST_PROCESSING

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class EntropyCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on entropy.

        This criterion computes the mean entropy of the predicted probability distribution.
        Higher entropy values indicate greater uncertainty.

        .. math::
            H(\mathbf{p}) = -\sum_{i=1}^{C} p_i \log(p_i)

        where :math:`\mathbf{p} = [p_1, p_2, \dots, p_C]` is the probability vector.

        Attributes:
            input_type: Expected input type is estimated probabilities.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the entropy of the predicted probability distribution.

        Args:
            inputs: Tensor of estimated probabilities with shape (batch_size, num_classes).

        Returns:
            Tensor: Mean entropy value for each sample.
        """
        return torch.special.entr(inputs).sum(dim=-1).mean(dim=1)


class MutualInformationCriterion(TUOODCriterion):
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on mutual information (BALD).

        This criterion computes the mutual information between the prediction and the
        model parameters across the ensemble's predictions — a classical estimator of
        *epistemic* uncertainty. Higher mutual information values indicate greater
        epistemic uncertainty and thus a higher likelihood of being out-of-distribution.

        Given ensemble predictions :math:`\{\mathbf{p}^{(k)}\}_{k=1}^{K}`, the mutual information is

        .. math::
            I(y, \theta) = H\!\left(\frac{1}{K}\sum_{k=1}^{K} \mathbf{p}^{(k)}\right)
            - \frac{1}{K}\sum_{k=1}^{K} H(\mathbf{p}^{(k)}),

        i.e. the total predictive entropy minus the average per-estimator entropy.

        Attributes:
            ensemble_only: Requires ensemble predictions.
            input_type: Expected input type is estimated probabilities.
        """
        super().__init__()
        self.mi_metric = MutualInformation(reduction="none")

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute mutual information from ensemble predictions.

        Args:
            inputs: Tensor of ensemble probabilities with shape
                (ensemble_size, batch_size, num_classes).

        Returns:
            Tensor: Mutual information for each sample.
        """
        return self.mi_metric(inputs)


class VariationRatioCriterion(TUOODCriterion):
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on variation ratio.

        This criterion computes the variation ratio from ensemble predictions.
        Higher variation ratio values indicate greater uncertainty.

        Given ensemble predictions where :math:`n_{\text{mode}}` is the count of the most frequently
        predicted class among :math:`K` predictions, the variation ratio is computed as:

        .. math::
            \text{VR} = 1 - \frac{n_{\text{mode}}}{K}

        Attributes:
            ensemble_only: Requires ensemble predictions.
            input_type: Expected input type is estimated probabilities.
        """
        super().__init__()
        self.vr_metric = VariationRatio(reduction="none", probabilistic=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute variation ratio from ensemble predictions.

        Args:
            inputs: Tensor of ensemble probabilities with shape
                (ensemble_size, batch_size, num_classes).

        Returns:
            Tensor: Variation ratio for each sample.
        """
        return self.vr_metric(inputs.transpose(0, 1))


def get_ood_criterion(ood_criterion: type[TUOODCriterion] | TUOODCriterion | str) -> TUOODCriterion:
    """Get an OOD criterion instance based on a string identifier or class type.

    Args:
        ood_criterion: A string identifier for a predefined OOD criterion
            or a subclass of `TUOODCriterion`.

    Returns:
        TUOODCriterion: An instance of the requested OOD criterion.

    Raises:
        ValueError: If the input string or class type is invalid.
    """
    if isinstance(ood_criterion, str):
        if ood_criterion == "logit":
            return MaxLogitCriterion()
        if ood_criterion == "energy":
            return EnergyCriterion()
        if ood_criterion == "msp":
            return MaxSoftmaxCriterion()
        if ood_criterion == "post_processing":
            return PostProcessingCriterion()
        if ood_criterion == "deup":
            return DEUPCriterion()
        if ood_criterion == "entropy":
            return EntropyCriterion()
        if ood_criterion == "mutual_information":
            return MutualInformationCriterion()
        if ood_criterion == "variation_ratio":
            return VariationRatioCriterion()
        raise ValueError(
            "The OOD criterion must be one of 'msp', 'logit', 'energy', 'entropy',"
            " 'mutual_information', 'variation_ratio', or 'post_processing', or 'deup'. "
            f"Got {ood_criterion=}."
        )
    if isinstance(ood_criterion, type):
        return ood_criterion()
    if isinstance(ood_criterion, TUOODCriterion):
        return ood_criterion
    raise ValueError(
        f"The OOD criterion should be a string or a subclass of TUOODCriterion. Got {ood_criterion}."
    )
