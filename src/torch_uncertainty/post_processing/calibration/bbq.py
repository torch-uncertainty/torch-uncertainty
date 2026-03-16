import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import PostProcessing

from .utils import _determine_dimensionality, _extract_data


class BBQScaler(PostProcessing):
    num_classes: int | None = None
    bbq_models: list[tuple[list[dict], Tensor]]

    def __init__(
        self,
        model: nn.Module | None = None,
        max_bins: int = 15,
        prior_weight: float = 2.0,
        model_pruning: float | None = 1e-9,
        eps: float = 1e-6,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Bayesian Binning into Quantiles (BBQ) post-processing.

        BBQ is a non-parametric calibration method that extends histogram
        binning by considering multiple equal-frequency binning models
        (differing in the number of bins). It computes a weighted average of
        the predictions from all models, where the weights are the posterior
        probabilities of the binning models given the calibration data.

        For multiclass inputs, this scaler strictly utilizes a One-vs-Rest (OvR)
        strategy, fitting independent Bayesian ensembles per class.

        Args:
            model (nn.Module | None): Model to calibrate. Defaults to ``None``.
            max_bins (int): The maximum number of bins to consider. The scaler
                will evaluate all binning schemes from 2 up to ``max_bins``.
                Defaults to ``15``.
            prior_weight (float): The equivalent sample size ($N'$) for the
                uniform prior distributed across bins to penalize models with
                too many bins. Defaults to ``2.0``(the value used in the original)
                paper.
            model_pruning (float | None): Prune a model if its weight is below :attr:`model_pruning`.
                Do not prune if ``None``. Defaults to ``1e-9``.
            eps (float): Small value for stability when converting probs back
                to logits. Defaults to ``1e-6``.
            device (Literal["cpu", "cuda"]]= | torch.device | None): Device to use
                for tensor operations. Defaults to ``None``.

        References:
            [1] Obtaining Well Calibrated Probabilities Using Bayesian Binning.
            In AAAI 2015.
            <https://www.dbmi.pitt.edu/wp-content/uploads/2022/10/Obtaining-well-calibrated-probabilities-using-Bayesian-binning.pdf>`_.

        Remark:
            This implementation will work better with a limited number of classes.
            Otherwise, the equal-frequency bins will be imprecise for high-confidence
            values.
        """
        super().__init__(model)

        if max_bins < 2:
            raise ValueError(f"max_bins must be at least 2. Got {max_bins}.")

        self.max_bins = max_bins
        self.prior_weight = prior_weight
        self.eps = eps
        self.device = device
        self.model_pruning = model_pruning

    def fit(
        self,
        dataloader: DataLoader,
        progress: bool = True,
    ) -> None:
        """Fit the BBQ ensembles to the calibration data.

        Args:
            dataloader (DataLoader): Dataloader providing the calibration data.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to ``True``.
        """
        if self.model is None or isinstance(self.model, nn.Identity):
            logging.warning(
                "model is None. Fitting post_processing method on the dataloader's data directly."
            )
            self.model = nn.Identity()

        all_logits, all_labels = _extract_data(
            dataloader=dataloader, model=self.model, device=self.device, progress=progress
        )
        self.num_classes, probs, labels = _determine_dimensionality(all_logits, all_labels)

        self.bbq_models = []
        labels_one_hot = (
            F.one_hot(labels.long(), self.num_classes).float() if self.num_classes > 1 else None
        )

        for c in range(self.num_classes):
            class_probs = probs[:, c] if self.num_classes > 1 else probs
            class_labels = labels_one_hot[:, c] if labels_one_hot is not None else labels

            class_models, log_scores, seen_edges = [], [], set()
            for b in range(2, self.max_bins + 1):
                # Define equal-frequency quantiles
                quantiles = torch.linspace(0.0, 1.0, b + 1, device=self.device)
                edges = torch.quantile(class_probs, quantiles)

                # Use unique to handle degenerate distributions with many identical probabilities
                edges = torch.unique(edges)
                if len(edges) < 2:
                    continue

                # Ensure edges span the entire [0, 1] interval perfectly
                edges[0] = -1e-5
                edges[-1] = 1.0 + 1e-5

                # Skip duplicate binning models that arise from `unique` collapsing quantiles
                edges_tuple = tuple(edges.cpu().numpy().round(9))
                if edges_tuple in seen_edges:
                    continue
                seen_edges.add(edges_tuple)

                # Bucketize and count
                indices = torch.bucketize(class_probs, edges) - 1
                indices = torch.clamp(indices, 0, len(edges) - 2)

                # Compute log-marginal likelihood P(D|M)
                bin_pos_freq, log_score = self.compute_log_score(
                    indices=indices, edges=edges, class_labels=class_labels
                )

                class_models.append({"edges": edges, "bin_pos_freq": bin_pos_freq})
                log_scores.append(log_score)

            # Fallback for entirely degenerate inputs
            if not class_models:
                logging.warning("BBQScaler: entirely degenerate inputs")
                class_models.append(
                    {
                        "edges": torch.tensor([-1e-5, 1.0 - 1e-5], device=self.device),
                        "bin_pos_freq": torch.tensor([class_labels.mean()], device=self.device),
                    }
                )
                log_scores.append(torch.tensor(0.0, device=self.device))

            # Compute posterior weights P(M|D) using Softmax
            weights = F.softmax(torch.stack(log_scores), dim=0)
            if self.model_pruning is None:
                pruned_models = class_models
                pruned_weights = weights
            else:
                kept_indices = weights >= self.model_pruning
                if kept_indices.sum() == 0:  # coverage: ignore
                    raise ValueError(
                        "BBQScaler: Error while pruning models. Lower the value of the pruning argument."
                    )
                pruned_weights = weights[kept_indices]
                pruned_models = [
                    m for m, keep in zip(class_models, kept_indices.cpu(), strict=True) if keep
                ]

            self.bbq_models.append((pruned_models, pruned_weights))

        self.trained = True

    def compute_log_score(self, indices: Tensor, edges: Tensor, class_labels: Tensor):
        actual_bins = len(edges) - 1
        num_samples_per_bin = torch.bincount(indices, minlength=actual_bins).float()
        num_positive_samples_per_bin = torch.bincount(
            indices, weights=class_labels, minlength=actual_bins
        ).float()
        num_negative_samples_per_bin = num_samples_per_bin - num_positive_samples_per_bin

        prior_base_term = torch.tensor(self.prior_weight / actual_bins, device=self.device)
        pb = (edges[:-1] + edges[1:]) / 2
        alpha = prior_base_term * pb
        beta = prior_base_term * (1 - pb)

        # Following the formula from page 2
        log_marg = (
            torch.lgamma(prior_base_term)
            - torch.lgamma(num_samples_per_bin + prior_base_term)
            + torch.lgamma(num_positive_samples_per_bin + alpha)
            - torch.lgamma(alpha)
            + torch.lgamma(num_negative_samples_per_bin + beta)
            - torch.lgamma(beta)
        )
        log_score = log_marg.sum()

        # Save model expectations
        bin_pos_freq = num_positive_samples_per_bin / num_samples_per_bin
        return bin_pos_freq, log_score

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply Bayesian Binning into Quantiles and return calibrated logits."""
        if self.model is None or not self.trained:
            logging.warning("Scaler not trained. Returning raw inputs.")
            return self.model(inputs)

        logits = self.model(inputs)
        calib_probs_list = []

        probs = (
            torch.sigmoid(logits).flatten()
            if self.num_classes == 1
            else torch.softmax(logits, dim=-1)
        )
        # Vectorized evaluation per class
        for c in range(self.num_classes):
            class_probs = probs[:, c] if self.num_classes != 1 else probs
            class_calib = torch.zeros_like(class_probs)

            class_models, weights = self.bbq_models[c]

            for weight, class_model in zip(weights, class_models, strict=True):
                edges = class_model["edges"]
                bin_pos_freq = class_model["bin_pos_freq"]

                indices = torch.bucketize(class_probs, edges) - 1
                indices = torch.clamp(indices, 0, len(bin_pos_freq) - 1)

                class_calib += weight * bin_pos_freq[indices]

            calib_probs_list.append(class_calib)

        if self.num_classes == 1:
            calib_probs = calib_probs_list[0]
        else:
            calib_probs = torch.stack(calib_probs_list, dim=-1)
            # Normalize to ensure multiclass probabilities sum to 1
            calib_probs /= calib_probs.sum(dim=-1, keepdim=True)

        # Convert back to pseudo-logit space
        calib_probs = calib_probs.clamp(self.eps, 1 - self.eps)
        if self.num_classes == 1:
            return torch.logit(calib_probs, eps=self.eps)
        return torch.log(calib_probs)
