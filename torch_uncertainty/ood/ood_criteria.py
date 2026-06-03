import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
from torch import Tensor, nn
from tqdm import tqdm

from torch_uncertainty.metrics import MutualInformation, VariationRatio
from torch_uncertainty.ood.nets import (
    AdaScaleANet,
    ASHNet,
    ReactNet,
    ScaleNet,
)
from torch_uncertainty.ood.nets.backbone_utils import (
    forward_with_features,
    get_fc_numpy,
    get_feature_dim,
)
from torch_uncertainty.ood.utils import load_config

logger = logging.getLogger(__name__)


def normalizer(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class OODCriterionInputType(Enum):
    """Enum representing the type of input expected by the OOD (Out-of-Distribution) criteria.

    Attributes:
        LOGIT (int): Represents that the input is in the form of logits (pre-softmax values).
        PROB (int): Represents that the input is in the form of probabilities (post-softmax values).
        ESTIMATOR_PROB (int): Represents that the input is in the form of estimated probabilities
            from an ensemble or other probabilistic model.
    """

    LOGIT = 1
    PROB = 2
    ESTIMATOR_PROB = 3
    DATASET = 4


class TUOODCriterion(ABC, nn.Module):
    single_only = True
    input_type: OODCriterionInputType
    ensemble_only = False

    def __init__(self) -> None:
        """Abstract base class for Out-of-Distribution (OOD) criteria.

        This class defines a common interface for implementing various OOD detection
        criteria. Subclasses must implement the `forward` method.

        Attributes:
            input_type (OODCriterionInputType): Type of input expected by the criterion.
            ensemble_only (bool): Whether the criterion requires ensemble outputs.
        """
        super().__init__()
        self.setup_flag = False
        self.hyperparam_search_done = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass for the OOD criterion.

        Args:
            inputs (Tensor): The input tensor representing model outputs.

        Returns:
            Tensor: OOD score computed according to the criterion.
        """


class MaxLogitCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.LOGIT

    def __init__(self) -> None:
        """OOD criterion based on the maximum logit value.

        This criterion computes the negative of the highest logit value across
        the output dimensions. Lower maximum logits indicate greater uncertainty.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is logits.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative of the maximum logit value.

        Args:
            inputs (Tensor): Tensor of logits with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative of the maximum logit value for each sample.
        """
        return -inputs.mean(dim=1).max(dim=-1).values


class EnergyCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.LOGIT

    def __init__(self) -> None:
        r"""OOD criterion based on the energy function.

        This criterion computes the negative log-sum-exp of the logits.
        Higher energy values indicate greater uncertainty.

        .. math::
            E(\mathbf{z}) = -\log\left(\sum_{i=1}^{C} \exp(z_i)\right)

        where :math:`\mathbf{z} = [z_1, z_2, \dots, z_C]` is the logit vector.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is logits.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative energy score.

        Args:
            inputs (Tensor): Tensor of logits with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative energy score for each sample.
        """
        return -inputs.mean(dim=1).logsumexp(dim=-1)


class MaxSoftmaxCriterion(TUOODCriterion):
    single_only = False
    input_type = OODCriterionInputType.PROB

    def __init__(self) -> None:
        r"""OOD criterion based on maximum softmax probability.

        This criterion computes the negative of the highest softmax probability.
        Lower maximum probabilities indicate greater uncertainty.

        .. math::
            \text{score} = -\max_{i}(p_i)

        where :math:`\mathbf{p} = [p_1, p_2, \dots, p_C]` is the probability vector.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is probabilities.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the negative of the maximum softmax probability.

        Args:
            inputs (Tensor): Tensor of probabilities with shape (batch_size, num_classes).

        Returns:
            Tensor: Negative of the highest softmax probability for each sample.
        """
        return -inputs.max(-1)[0]


class EntropyCriterion(TUOODCriterion):
    single_only = False
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on entropy.

        This criterion computes the mean entropy of the predicted probability distribution.
        Higher entropy values indicate greater uncertainty.

        .. math::
            H(\mathbf{p}) = -\sum_{i=1}^{C} p_i \log(p_i)

        where :math:`\mathbf{p} = [p_1, p_2, \dots, p_C]` is the probability vector.

        Attributes:
            input_type (OODCriterionInputType): Expected input type is estimated probabilities.
        """
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the entropy of the predicted probability distribution.

        Args:
            inputs (Tensor): Tensor of estimated probabilities with shape (batch_size, num_classes).

        Returns:
            Tensor: Mean entropy value for each sample.
        """
        return torch.special.entr(inputs).sum(dim=-1).mean(dim=1)


class MutualInformationCriterion(TUOODCriterion):
    single_only = False
    ensemble_only = True
    input_type = OODCriterionInputType.ESTIMATOR_PROB

    def __init__(self) -> None:
        r"""OOD criterion based on mutual information.

        This criterion computes the mutual information between ensemble predictions.
        Higher mutual information values indicate lower uncertainty.

        Given ensemble predictions :math:`\{\mathbf{p}^{(k)}\}_{k=1}^{K}`, the mutual information is computed as:

        .. math::
            I(y, \theta) = H\Big(\frac{1}{K}\sum_{k=1}^{K} \mathbf{p}^{(k)}\Big) - \frac{1}{K}\sum_{k=1}^{K} H(\mathbf{p}^{(k)})

        Attributes:
            ensemble_only (bool): Requires ensemble predictions.
            input_type (OODCriterionInputType): Expected input type is estimated probabilities.
        """
        super().__init__()
        self.mi_metric = MutualInformation(reduction="none")

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute mutual information from ensemble predictions.

        Args:
            inputs (Tensor): Tensor of ensemble probabilities with shape
                (ensemble_size, batch_size, num_classes).

        Returns:
            Tensor: Mutual information for each sample.
        """
        return self.mi_metric(inputs)


class VariationRatioCriterion(TUOODCriterion):
    single_only = False
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
            ensemble_only (bool): Requires ensemble predictions.
            input_type (OODCriterionInputType): Expected input type is estimated probabilities.
        """
        super().__init__()
        self.vr_metric = VariationRatio(reduction="none", probabilistic=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute variation ratio from ensemble predictions.

        Args:
            inputs (Tensor): Tensor of ensemble probabilities with shape
                (ensemble_size, batch_size, num_classes).

        Returns:
            Tensor: Variation ratio for each sample.
        """
        return self.vr_metric(inputs.transpose(0, 1))


class ScaleCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the Scale method.

        This criterion uses a scaling-based approach to compute OOD scores.
        It applies a thresholding mechanism to the network's output and computes
        the energy confidence score. The lower the energy confidence, the higher
        the uncertainty.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            percentile (float): Percentile value used for thresholding.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = config.postprocessor.postprocessor_sweep

    def forward(self, net: nn.Module, data: Any) -> Tensor:
        net = ScaleNet(net)
        output = net.forward_threshold(data, self.percentile)
        energyconf = torch.logsumexp(output.data, dim=1)
        return -energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile


class ASHCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the ASH (Activation Shift) method.

        This criterion uses a thresholding mechanism to compute OOD scores
        based on the network's activations. It applies a percentile-based
        threshold to the activations and computes the energy confidence score.
        Lower energy confidence indicates higher uncertainty.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            percentile (float): Percentile value used for thresholding.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        net = ASHNet(net)
        output = net.forward_threshold(data, self.percentile)
        energyconf = torch.logsumexp(output.data, dim=1)
        return -energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile


class ReactCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the React (Rectified Activation) method.

        This criterion uses a thresholding mechanism to compute OOD scores
        based on the network's activations. It applies a percentile-based
        threshold to the activations and computes the energy confidence score.
        Lower energy confidence indicates higher uncertainty.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            percentile (float): Percentile value used for thresholding.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader, ood_loaders):
        if not self.setup_flag:
            activation_log = []
            net = ReactNet(net)
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader["val"], desc="Setup: ", position=0, leave=True):
                    data = batch[0].cuda().float()
                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
        else:
            pass

        self.threshold = np.percentile(self.activation_log.flatten(), self.percentile)

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        net = ReactNet(net)
        output = net.forward_threshold(data, self.threshold)
        energyconf = torch.logsumexp(output.data, dim=1)
        return -energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.threshold = np.percentile(self.activation_log.flatten(), self.percentile)
        logger.info(
            "Threshold at percentile %d over id data is: %s",
            self.percentile,
            self.threshold,
        )

    def get_hyperparam(self):
        return self.percentile


class AdaScaleCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the AdaScale method.

        This criterion uses an adaptive scaling approach to compute OOD scores.
        It applies a percentile-based thresholding mechanism to the network's
        features and computes the energy confidence score. Lower energy confidence
        indicates higher uncertainty.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            percentile (float): Percentile value used for thresholding.
            k1 (int): Number of top-k features considered for correction term.
            k2 (int): Number of top-k features considered for feature shift.
            lmbda (float): Scaling factor for feature shift.
            o (float): Fraction of pixels used for perturbation.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.k1 = self.args.k1
        self.k2 = self.args.k2
        self.lmbda = self.args.lmbda
        self.o = self.args.o
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader, ood_loaders):
        net = AdaScaleANet(net)
        if not self.setup_flag:
            feature_log = []
            feature_perturbed_log = []
            feature_shift_log = []
            net.eval()
            self.feature_dim = get_feature_dim(net.backbone)
            with torch.no_grad():
                for batch in tqdm(id_loader["val"], desc="Setup: ", position=0, leave=True):
                    data = batch[0].cuda().float()
                    with torch.enable_grad():
                        data.requires_grad = True
                        output, feature = net(data, return_feature=True)
                        labels = output.detach().argmax(dim=1)
                        net.zero_grad()
                        score = output[torch.arange(len(labels)), labels]
                        score.backward(torch.ones_like(labels))
                        grad = data.grad.data.detach()
                    feature_log.append(feature.data.cpu())
                    data_perturbed = self.perturb(data, grad)
                    _, feature_perturbed = net(data_perturbed, return_feature=True)
                    feature_shift = abs(feature - feature_perturbed)
                    feature_shift_log.append(feature_shift.data.cpu())
                    feature_perturbed_log.append(feature_perturbed.data.cpu())
            all_features = torch.cat(feature_log, axis=0)
            all_perturbed = torch.cat(feature_perturbed_log, axis=0)
            all_shifts = torch.cat(feature_shift_log, axis=0)

            total_samples = all_features.size(0)
            num_samples = (
                self.args.num_samples if hasattr(self.args, "num_samples") else total_samples
            )
            indices = torch.randperm(total_samples)[:num_samples]

            self.feature_log = all_features[indices]
            self.feature_perturbed_log = all_perturbed[indices]
            self.feature_shift_log = all_shifts[indices]
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def get_percentile(self, feature, feature_perturbed, feature_shift):
        topk_indices = torch.topk(feature, dim=1, k=self.k1_)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(feature_perturbed), 1, topk_indices
        )  # correction term C_o
        topk_indices = torch.topk(feature, dim=1, k=self.k2_)[1]
        topk_feature_shift = torch.gather(feature_shift, 1, topk_indices)  # Q
        topk_norm = topk_feature_perturbed.sum(dim=1) + self.lmbda * topk_feature_shift.sum(
            dim=1
        )  # Q^{\prime}
        percent = 1 - self.ecdf(topk_norm.cpu())
        percentile = self.min_percentile + percent * (self.max_percentile - self.min_percentile)
        return torch.from_numpy(percentile)

    @torch.no_grad()
    def forward(self, net: nn.Module, data):
        net = AdaScaleANet(net)
        with torch.enable_grad():
            data.requires_grad = True
            output, feature = net(data, return_feature=True)
            labels = output.detach().argmax(dim=1)
            net.zero_grad()
            score = output[torch.arange(len(labels)), labels]
            score.backward(torch.ones_like(labels))
            grad = data.grad.data.detach()
            data.requires_grad = False
        data_perturbed = self.perturb(data, grad)
        _, feature_perturbed = net(data_perturbed, return_feature=True)
        feature_shift = abs(feature - feature_perturbed)
        percentile = self.get_percentile(feature, feature_perturbed, feature_shift)
        output = net.forward_threshold(feature, percentile)
        conf = torch.logsumexp(output, dim=1)
        return -conf

    @torch.no_grad()
    def perturb(self, data, grad):
        batch_size, channels, height, width = data.shape
        n_pixels = int(channels * height * width * self.o)
        abs_grad = abs(grad).view(batch_size, channels * height * width)
        _, topk_indices = torch.topk(abs_grad, n_pixels, dim=1, largest=False)
        mask = torch.zeros_like(abs_grad, dtype=torch.uint8)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.view(batch_size, channels, height, width)
        return data + grad.sign() * mask * 0.5

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.min_percentile, self.max_percentile = self.percentile[0], self.percentile[1]
        self.k1 = hyperparam[1]
        self.k2 = hyperparam[2]
        self.lmbda = hyperparam[3]
        self.o = hyperparam[4]
        self.k1_ = int(self.feature_dim * self.k1 / 100)
        self.k2_ = int(self.feature_dim * self.k2 / 100)
        topk_indices = torch.topk(self.feature_log, k=self.k1_, dim=1)[1]
        topk_feature_perturbed = torch.gather(
            torch.relu(self.feature_perturbed_log), 1, topk_indices
        )
        topk_indices = torch.topk(self.feature_log, k=self.k2_, dim=1)[1]
        topk_feature_shift_log = torch.gather(self.feature_shift_log, 1, topk_indices)
        sum_log = topk_feature_perturbed.sum(dim=1) + self.lmbda * topk_feature_shift_log.sum(dim=1)
        self.ecdf = ECDF(sum_log)

    def get_hyperparam(self):
        return [self.percentile, self.k1, self.k2, self.lmbda, self.o]


class VIMCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the VIM (Variance-Informed Mahalanobis) method.

        This criterion uses a Mahalanobis distance-based approach to compute OOD scores.
        It extracts features from the in-distribution training data, computes the
        empirical covariance matrix, and projects the features onto a subspace
        defined by the top eigenvectors. The OOD score is computed as a combination
        of the variance-informed Mahalanobis distance and the energy score.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            dim (int): Number of dimensions to retain in the subspace projection.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.args_dict = config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim

    def setup(self, net: nn.Module, id_loader, ood_loaders):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                self.w, self.b = get_fc_numpy(net)
                logger.info("Extracting in-distribution training features")
                feature_id_train = []
                for batch in tqdm(id_loader["train"], desc="Setup: ", position=0, leave=True):
                    data = batch[0].cuda().float()
                    _, feature = forward_with_features(net, data)
                    feature_id_train.append(feature.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim :]]).T
            )

            vlogit_id_train = norm(np.matmul(feature_id_train - self.u, self.NS), axis=-1)
            self.alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
            logger.info("Computed alpha: %.4f", self.alpha)

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        _, feature_ood = forward_with_features(net, data)
        feature_ood = feature_ood.cpu()
        logit_ood = feature_ood @ self.w.T + self.b
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS), axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return -torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim


class ODINCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the ODIN (Out-of-Distribution Detector for Neural Networks) method.

        This criterion uses temperature scaling and input perturbations to compute OOD scores.
        It applies a small perturbation to the input data based on the gradient of the cross-entropy
        loss with respect to the input. The confidence score is then calculated using the perturbed
        input and temperature-scaled logits. Lower confidence scores indicate higher uncertainty.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            temperature (float): Temperature scaling factor for logits.
            noise (float): Magnitude of the input perturbation.
            input_std (list): Standard deviation values for input normalization.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args

        self.temperature = 1
        self.noise = 0.0014
        try:
            self.input_std = [0.2470, 0.2435, 0.2616]  # // to change
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]
        self.args_dict = config.postprocessor.postprocessor_sweep

    def forward(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = net(data)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        temp_inputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = net(temp_inputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nn_output = output.detach()
        nn_output = nn_output - nn_output.max(dim=1, keepdims=True).values
        nn_output = nn_output.exp() / nn_output.exp().sum(dim=1, keepdims=True)

        conf, _ = nn_output.max(dim=1)

        return -conf

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]


class NECOCriterion(TUOODCriterion):
    """OOD criterion based on Neural Collapse (NECO).

    Fits a StandardScaler and PCA on in-distribution training features, then scores
    samples by the ratio of norm in the leading PCA subspace to the full feature norm.
    Higher scores indicate in-distribution samples. Heavily inspired by:

    NECO: NEural Collapse Based Out-of-distribution detection (Ammar et al., 2023).
    https://arxiv.org/abs/2310.06823
    """

    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.neco_dim = self.args.neco_dim
        self.use_scaler = getattr(self.args, "use_scaler", True)
        self.scale_by_maxlogit = getattr(self.args, "scale_by_maxlogit", True)
        self.args_dict = config.postprocessor.postprocessor_sweep
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        if "train" not in id_loader_dict:
            raise RuntimeError(
                "NECO requires an in-distribution train loader in setup. "
                "Ensure the datamodule exposes train_dataloader()."
            )

        net.eval()
        device = next(net.parameters()).device
        activation_log = []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict["train"], desc="Setup: ", position=0, leave=True):
                data = batch[0].to(device).float()
                _, feature = forward_with_features(net, data)
                activation_log.append(feature.cpu().numpy())

        train_feats = np.concatenate(activation_log, axis=0)
        if self.use_scaler:
            self.scaler = StandardScaler()
            train_scaled = self.scaler.fit_transform(train_feats)
        else:
            self.scaler = None
            train_scaled = train_feats

        n_components = min(train_scaled.shape[0], train_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(train_scaled)
        self.neco_dim = min(self.neco_dim, n_components)
        self.setup_flag = True

    def _ratio_score(self, features: np.ndarray, logits: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            features = self.scaler.transform(features)
        n_dims = min(self.neco_dim, self.pca.n_components_)
        reduced = self.pca.transform(features)[:, :n_dims]
        norm_full = norm(features, axis=-1)
        norm_red = norm(reduced, axis=-1)
        ratio = norm_red / (norm_full + 1e-10)
        if self.scale_by_maxlogit:
            ratio = ratio * logits.max(axis=-1)
        return ratio

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        logits, feature = forward_with_features(net, data)
        feat_np = feature.cpu().numpy()
        log_np = logits.cpu().numpy()
        scores = self._ratio_score(feat_np, log_np)
        return -torch.from_numpy(scores).to(device=feature.device, dtype=feature.dtype)

    def set_hyperparam(self, hyperparam: list):
        if self.pca is not None:
            self.neco_dim = min(hyperparam[0], self.pca.n_components_)
        else:
            self.neco_dim = hyperparam[0]

    def get_hyperparam(self):
        return self.neco_dim


class KNNCriterion(TUOODCriterion):
    """OOD criterion based on the K-Nearest Neighbors (KNN) method.

    This criterion uses a KNN-based approach to compute OOD scores. It builds a feature
    bank from the in-distribution training data and calculates the distance of test
    samples to their K-th nearest neighbor in the feature space. Lower distances
    indicate higher confidence, while higher distances indicate greater uncertainty.

    Heavily inspired by the OpenOOD repository:
    https://github.com/Jingkang50/OpenOOD

    Attributes:
        input_type (OODCriterionInputType): Expected input type is dataset.
        K (int): Number of nearest neighbors to consider.
        activation_log (np.ndarray): Log of activations from the in-distribution training data.
        args_dict (dict): Dictionary containing hyperparameter sweep configurations.
    """

    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict["train"], desc="Setup: ", position=0, leave=True):
                    data = batch[0].cuda().float()
                    _, feature = forward_with_features(net, data)
                    activation_log.append(normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(feature.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        _, feature = forward_with_features(net, data)
        feature_normed = normalizer(feature.data.cpu().numpy())
        dis, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -dis[:, -1]
        return -torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K


class GENCriterion(TUOODCriterion):
    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        """OOD criterion based on the Generalized Entropy (GEN) method.

        This criterion uses a generalized entropy-based approach to compute OOD scores.
        It applies a power transformation to the top-M softmax probabilities and computes
        the generalized entropy score. Lower scores indicate higher uncertainty.

        Heavily inspired by the OpenOOD repository:
        https://github.com/Jingkang50/OpenOOD

        Attributes:
            input_type (OODCriterionInputType): Expected input type is dataset.
            gamma (float): Power transformation parameter for generalized entropy.
            m (int): Number of top-M probabilities considered for the computation.
            args_dict (dict): Dictionary containing hyperparameter sweep configurations.
        """
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.gamma = self.args.gamma
        self.m = self.args.m
        self.args_dict = config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf = self.generalized_entropy(score, self.gamma, self.m)
        return -conf

    def set_hyperparam(self, hyperparam: list):
        self.gamma = hyperparam[0]
        self.m = hyperparam[1]

    def get_hyperparam(self):
        return [self.gamma, self.m]

    def generalized_entropy(self, softmax_id_val, gamma=0.1, m=100):
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -m:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), dim=1)

        return -scores


def knn_score(bankfeas, queryfeas, k=100, use_min=False):
    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    dist, _ = index.search(queryfeas, k)
    return np.array(dist.use_min(axis=1)) if use_min else np.array(dist.mean(axis=1))


class NNGuideCriterion(TUOODCriterion):
    """NNGuideCriterion is a criterion for out-of-distribution (OOD) detection
    that utilizes nearest neighbor guidance based on features and logits
    extracted from a neural network. This class is heavily inspired by the
    OpenOOD repository: https://github.com/Jingkang50/OpenOOD.

    Attributes:
        input_type (OODCriterionInputType): Specifies the type of input for the criterion.
        args (Namespace): Arguments related to the postprocessor configuration.
        K (int): Number of nearest neighbors to consider for the k-NN score.
        alpha (float): Fraction of the in-distribution training data to use for setup.
        activation_log (Any): Placeholder for activation logs (currently unused).
        args_dict (dict): Dictionary of postprocessor sweep arguments.
        setup_flag (bool): Indicates whether the setup process has been completed.
        bank_guide (np.ndarray): Precomputed guidance bank combining features and confidence scores.

    Methods:
        setup(net, id_loader_dict, ood_loader_dict):
            Prepares the guidance bank using in-distribution training data.

        forward(net, data):
            Computes the OOD score for the given data using the guidance bank.

        set_hyperparam(hyperparam):
            Sets the hyperparameters K and alpha.

        get_hyperparam():
            Retrieves the current hyperparameters K and alpha.
    """

    input_type = OODCriterionInputType.DATASET

    def __init__(self, config) -> None:
        super().__init__()
        self.args = config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.alpha = self.args.alpha
        self.activation_log = None
        self.args_dict = config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            bank_feas = []
            bank_logits = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict["train"], desc="Setup: ", position=0, leave=True):
                    data = batch[0].cuda().float()

                    logit, feature = forward_with_features(net, data)
                    bank_feas.append(normalizer(feature.data.cpu().numpy()))
                    bank_logits.append(logit.data.cpu().numpy())
                    if len(bank_feas) * id_loader_dict["train"].batch_size > int(
                        len(id_loader_dict["train"].dataset) * self.alpha
                    ):
                        break

            bank_feas = np.concatenate(bank_feas, axis=0)
            bank_confs = logsumexp(np.concatenate(bank_logits, axis=0), axis=-1)
            self.bank_guide = bank_feas * bank_confs[:, None]

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def forward(self, net: nn.Module, data: Any):
        logit, feature = forward_with_features(net, data)
        feas_norm = normalizer(feature.data.cpu().numpy())
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)

        conf = knn_score(self.bank_guide, feas_norm, k=self.K)
        score = conf * energy

        return -torch.from_numpy(score)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]
        self.alpha = hyperparam[1]

    def get_hyperparam(self):
        return [self.K, self.alpha]


def get_ood_criterion(ood_criterion):
    """Get an OOD criterion instance based on a string identifier or class type.

    Args:
        ood_criterion (str or type): A string identifier for a predefined OOD criterion
            or a subclass of `TUOODCriterion`.

    Returns:
        TUOODCriterion: An instance of the requested OOD criterion.

    Raises:
        ValueError: If the input string or class type is invalid.
    """
    config_dir = Path(__file__).parent / "configs"
    if isinstance(ood_criterion, str):
        if ood_criterion not in [
            "logit",
            "energy",
            "msp",
            "entropy",
            "mutual_information",
            "variation_ratio",
        ]:
            config_path = config_dir / f"{ood_criterion}.yml"
            if not config_path.is_file():
                raise ValueError(
                    f"No configuration file found for OOD criterion '{ood_criterion}'. "
                    f"Expected {config_path!r}."
                )
            config = load_config(str(config_path))
        if ood_criterion == "logit":
            return MaxLogitCriterion()
        if ood_criterion == "energy":
            return EnergyCriterion()
        if ood_criterion == "msp":
            return MaxSoftmaxCriterion()
        if ood_criterion == "entropy":
            return EntropyCriterion()
        if ood_criterion == "mutual_information":
            return MutualInformationCriterion()
        if ood_criterion == "variation_ratio":
            return VariationRatioCriterion()
        if ood_criterion == "scale":
            return ScaleCriterion(config)
        if ood_criterion == "ash":
            return ASHCriterion(config)
        if ood_criterion == "react":
            return ReactCriterion(config)
        if ood_criterion == "adascale_a":
            return AdaScaleCriterion(config)
        if ood_criterion == "vim":
            return VIMCriterion(config)
        if ood_criterion == "odin":
            return ODINCriterion(config)
        if ood_criterion == "knn":
            return KNNCriterion(config)
        if ood_criterion == "neco":
            return NECOCriterion(config)
        if ood_criterion == "gen":
            return GENCriterion(config)
        if ood_criterion == "nnguide":
            return NNGuideCriterion(config)
        raise ValueError(
            "The OOD criterion must be one of 'msp', 'logit', 'energy', 'entropy',"
            f" 'mutual_information' or 'variation_ratio'. Got {ood_criterion}."
        )
    if isinstance(ood_criterion, type) and issubclass(ood_criterion, TUOODCriterion):
        return ood_criterion()
    raise ValueError(
        f"The OOD criterion should be a string or a subclass of TUOODCriterion. Got {type(ood_criterion)}."
    )
