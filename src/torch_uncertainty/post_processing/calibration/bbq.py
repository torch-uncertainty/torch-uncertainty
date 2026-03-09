import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_uncertainty.post_processing import PostProcessing


class BBQScaler(PostProcessing):
    def __init__(
        self,
        model: nn.Module | None = None,
        max_bins: int = 15,
        prior_weight: float = 2.0,
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
            model (nn.Module): Model to calibrate. Defaults to ``None``.
            max_bins (int): The maximum number of bins to consider. The scaler
                will evaluate all binning schemes from 2 up to ``max_bins``.
                Defaults to ``15``.
            prior_weight (float): The equivalent sample size ($N'$) for the
                uniform prior distributed across bins to penalize models with
                too many bins. Defaults to ``2.0``.
            eps (float): Small value for stability when converting probs back
                to logits. Defaults to ``1e-6``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use
                for tensor operations. Defaults to ``None``.

        References:
            [1] Obtaining Well Calibrated Probabilities Using Bayesian Binning.
            In AAAI 2015.
            <https://www.dbmi.pitt.edu/wp-content/uploads/2022/10/Obtaining-well-calibrated-probabilities-using-Bayesian-binning.pdf>`_.
        """
        super().__init__(model)

        if max_bins < 2:
            raise ValueError(f"max_bins must be at least 2. Got {max_bins}.")

        self.max_bins = max_bins
        self.prior_weight = prior_weight
        self.eps = eps
        self.device = device

        self.num_classes: int | None = None
        self.models: list[tuple[list[dict], Tensor]] = []

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

        all_logits, all_labels = self._extract_data(dataloader, progress)

        # Determine dimensionality
        if all_logits.dim() == 1 or (all_logits.dim() == 2 and all_logits.shape[1] == 1):
            probs = torch.sigmoid(all_logits).flatten()
            labels = all_labels.float().flatten()
            self.num_classes = 1
        else:
            probs = torch.softmax(all_logits, dim=-1)
            labels = all_labels
            self.num_classes = probs.shape[1]

        self.models = []
        labels_one_hot = (
            F.one_hot(labels.long(), self.num_classes).float() if self.num_classes > 1 else None
        )

        # OvR Fitting process
        for c in range(self.num_classes):
            c_probs = probs[:, c] if self.num_classes > 1 else probs
            c_labels = labels_one_hot[:, c] if self.num_classes > 1 else labels

            class_models = []
            log_scores = []
            seen_edges = set()

            for b in range(2, self.max_bins + 1):
                # 1. Define equal-frequency quantiles
                quantiles = torch.linspace(0.0, 1.0, b + 1, device=self.device)
                edges = torch.quantile(c_probs, quantiles)

                # Use unique to handle degenerate distributions with many identical probabilities
                edges = torch.unique(edges)
                if len(edges) < 2:
                    continue

                # Ensure edges span the entire [0, 1] interval perfectly
                edges[0] = -1e-5
                edges[-1] = 1.0 + 1e-5

                # Skip duplicate binning models that arise from `unique` collapsing quantiles
                edges_tuple = tuple(edges.cpu().numpy().round(5))
                if edges_tuple in seen_edges:
                    continue
                seen_edges.add(edges_tuple)

                # 2. Bucketize and count
                indices = torch.bucketize(c_probs, edges) - 1
                indices = torch.clamp(indices, 0, len(edges) - 2)

                actual_bins = len(edges) - 1
                N_b = torch.bincount(indices, minlength=actual_bins).float()  # noqa: N806
                m_b = torch.bincount(indices, weights=c_labels, minlength=actual_bins).float()
                n_b = N_b - m_b

                # 3. Compute log-marginal likelihood P(D|M)
                alpha = self.prior_weight / (2 * actual_bins)
                beta = self.prior_weight / (2 * actual_bins)
                alpha_t = torch.tensor(alpha, device=self.device)
                beta_t = torch.tensor(beta, device=self.device)
                prior_term = torch.tensor(self.prior_weight / actual_bins, device=self.device)

                log_marg = (
                    torch.lgamma(prior_term)
                    - torch.lgamma(N_b + prior_term)
                    + torch.lgamma(m_b + alpha_t)
                    + torch.lgamma(n_b + beta_t)
                    - torch.lgamma(alpha_t)
                    - torch.lgamma(beta_t)
                )
                log_score = log_marg.sum()

                # 4. Save model expectations
                theta = (m_b + alpha_t) / (N_b + alpha_t + beta_t)

                class_models.append({"edges": edges, "theta": theta})
                log_scores.append(log_score)

            # Fallback for entirely degenerate inputs
            if not class_models:
                class_models.append(
                    {
                        "edges": torch.tensor([-1e-5, 1.0 + 1e-5], device=self.device),
                        "theta": torch.tensor([c_labels.mean()], device=self.device),
                    }
                )
                log_scores.append(torch.tensor(0.0, device=self.device))

            # Compute posterior weights P(M|D) using Softmax
            weights = F.softmax(torch.stack(log_scores), dim=0)
            self.models.append((class_models, weights))

        self.trained = True

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply Bayesian Binning into Quantiles and return calibrated logits."""
        if self.model is None or not self.trained:
            logging.warning("Scaler not trained. Returning raw inputs.")
            return self.model(inputs)

        logits = self.model(inputs)
        calib_probs_list = []

        # Vectorized inference per class
        for c in range(self.num_classes):
            c_probs = (
                torch.sigmoid(logits).flatten()
                if self.num_classes == 1
                else torch.softmax(logits, dim=-1)[:, c]
            )
            c_calib = torch.zeros_like(c_probs)

            class_models, weights = self.models[c]

            for weight, model in zip(weights, class_models, strict=True):
                edges = model["edges"]
                theta = model["theta"]

                indices = torch.bucketize(c_probs, edges) - 1
                indices = torch.clamp(indices, 0, len(theta) - 1)

                c_calib += weight * theta[indices]

            calib_probs_list.append(c_calib)

        if self.num_classes == 1:
            calib_probs = calib_probs_list[0]
        else:
            calib_probs = torch.stack(calib_probs_list, dim=-1)
            # Normalize to ensure multiclass probabilities sum to 1
            calib_probs /= calib_probs.sum(dim=-1, keepdim=True) + 1e-12

        # Convert back to pseudo-logit space
        calib_probs = calib_probs.clamp(self.eps, 1 - self.eps)
        if self.num_classes == 1:
            return torch.logit(calib_probs, eps=self.eps)
        return torch.log(calib_probs)

    def _extract_data(self, dataloader: DataLoader, progress: bool) -> tuple[Tensor, Tensor]:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, disable=not progress):
                logits = self.model(inputs.to(self.device))
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(self.device)
        all_labels = torch.cat(all_labels).to(self.device)

        if all_labels.ndim == 1:
            all_labels = all_labels.long()

        return all_logits, all_labels
