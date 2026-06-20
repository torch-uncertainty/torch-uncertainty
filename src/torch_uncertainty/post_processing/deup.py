"""Direct Epistemic Uncertainty Prediction (DEUP) post-processing.

Minimal PyTorch implementation of Lahlou et al. (2023) for TorchUncertainty
classification and regression routines. For sklearn/tabular/time-series DEUP,
see the standalone `deup` package: https://github.com/ursinasanderink/deup
"""

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import PostProcessing


class _ErrorPredictor(nn.Module):
    """Small MLP mapping features to non-negative predicted error."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


class DEUP(PostProcessing):
    def __init__(
        self,
        task: Literal["classification", "regression"],
        model: nn.Module | None = None,
        num_folds: int = 5,
        hidden_dim: int = 64,
        max_epochs: int = 40,
        lr: float = 1e-3,
        device: torch.device | str | None = None,
    ) -> None:
        """Direct Epistemic Uncertainty Prediction (DEUP).

        Trains an error predictor ``g`` on out-of-fold generalization errors collected
        from a calibration set, following Algorithm 2 in Lahlou et al. (2023).

        ``forward`` returns per-sample epistemic uncertainty estimates (non-negative).
        Pair with :class:`~torch_uncertainty.ood_criteria.DEUPCriterion` for OOD
        detection in :class:`~torch_uncertainty.routines.ClassificationRoutine`.

        Args:
            task: ``"classification"`` (per-sample cross-entropy error) or
                ``"regression"`` (squared error).
            model: Base model producing logits or point predictions.
            num_folds: Number of cross-validation folds for OOF error collection.
            hidden_dim: Hidden width of the error predictor MLP.
            max_epochs: Training epochs for each error-predictor fit.
            batch_size: Mini-batch size for error-predictor training. Defaults to ``256``.
            lr: Adam learning rate for the error predictor.
            device: Device for tensors and the error predictor.
            progress: Show progress bars during ``fit``.

        References:
            Lahlou et al. (2023). *DEUP: Direct Epistemic Uncertainty Prediction.*
            TMLR. https://openreview.net/forum?id=eGLdVRvvfQ

        Note:
            General-purpose / time-series DEUP (purged walk-forward, finance presets)
            lives in https://github.com/ursinasanderink/deup.
        """
        super().__init__(model=model)
        if task not in {"classification", "regression"}:
            raise ValueError(f"task must be 'classification' or 'regression'. Got {task}.")
        if num_folds < 2:
            raise ValueError(f"num_folds must be >= 2. Got {num_folds}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1. Got {hidden_dim}.")

        self.task = task
        self.num_folds = int(num_folds)
        self.hidden_dim = int(hidden_dim)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.device = torch.device(device or "cpu")

        self.error_predictor: _ErrorPredictor | None = None
        self._feature_dim: int | None = None

    def set_model(self, model: nn.Module) -> None:
        super().set_model(model)
        self.trained = False
        self.error_predictor = None

    def fit(self, dataloader: DataLoader) -> None:
        """Fit the error predictor on OOF errors from the calibration loader."""
        if self.model is None:
            raise RuntimeError("Model must be set before calling fit().")

        # ``fit`` is called from ``ClassificationRoutine.on_test_start``, which runs
        # inside the evaluation loop's ``torch.no_grad()``/``inference_mode`` context.
        # Training the error predictor needs autograd, so we re-enable it here and
        # leave inference mode to avoid producing inference tensors during feature
        # collection (which could not be used in the predictor's autograd graph).
        with torch.inference_mode(False), torch.enable_grad():
            features, errors = self._collect_features_and_errors(dataloader)
            oof_targets = self._out_of_fold_targets(features, errors)
            self._train_error_predictor(features, oof_targets)
        self.trained = True

    def forward(self, inputs: Tensor) -> Tensor:
        """Return epistemic uncertainty ``g(x) >= 0`` for each sample."""
        if self.model is None or self.error_predictor is None:
            raise RuntimeError("DEUP must be fitted before forward().")
        if not self.trained:
            logging.warning("DEUP has not been fitted; predictions may be unreliable.")

        self.error_predictor.eval()
        with torch.no_grad():
            feats = self._extract_features(inputs.to(self.device))
            return self.error_predictor(feats)

    def predict_proba(self, inputs: Tensor) -> Tensor:
        """Base-model probabilities (classification only; unchanged by DEUP)."""
        if self.model is None:
            raise RuntimeError("Model must be set.")
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification.")
        with torch.no_grad():
            logits = self.model(inputs.to(self.device))
            if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
                probs = logits.squeeze(-1).sigmoid()
                return torch.stack([1 - probs, probs], dim=-1)
            return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def _collect_features_and_errors(self, dataloader: DataLoader) -> tuple[Tensor, Tensor]:
        assert self.model is not None
        self.model.eval()
        feats_list: list[Tensor] = []
        err_list: list[Tensor] = []

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            feats = self._outputs_to_features(outputs)
            errs = self._per_sample_errors(outputs, labels)
            feats_list.append(feats)
            err_list.append(errs)

        features = torch.cat(feats_list, dim=0)
        errors = torch.cat(err_list, dim=0)
        self._feature_dim = features.shape[1]
        return features, errors

    def _out_of_fold_targets(self, features: Tensor, errors: Tensor) -> Tensor:
        num_samples = features.shape[0]
        oof = torch.zeros(num_samples, device=self.device)
        fold_sizes = [num_samples // self.num_folds] * self.num_folds
        for i in range(num_samples % self.num_folds):
            fold_sizes[i] += 1

        start = 0
        for fold_size in fold_sizes:
            val_idx = torch.arange(start, start + fold_size, device=self.device)
            train_mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)
            train_mask[val_idx] = False
            train_idx = train_mask.nonzero(as_tuple=True)[0]

            predictor = _ErrorPredictor(features.shape[1], self.hidden_dim).to(self.device)
            self._fit_predictor(
                predictor,
                features[train_idx],
                errors[train_idx],
            )
            predictor.eval()
            with torch.no_grad():
                oof[val_idx] = predictor(features[val_idx])
            start += fold_size
        return oof

    def _train_error_predictor(self, features: Tensor, targets: Tensor) -> None:
        self.error_predictor = _ErrorPredictor(features.shape[1], self.hidden_dim).to(self.device)
        self._fit_predictor(self.error_predictor, features, targets)

    def _fit_predictor(
        self,
        predictor: _ErrorPredictor,
        features: Tensor,
        targets: Tensor,
    ) -> None:
        predictor.train()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=self.lr)
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            pred = predictor(features)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()

    def _extract_features(self, inputs: Tensor) -> Tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return self._outputs_to_features(outputs)

    @staticmethod
    def _outputs_to_features(outputs: Tensor) -> Tensor:
        if outputs.dim() == 1:
            return outputs.unsqueeze(-1)
        if outputs.dim() == 2 and outputs.shape[1] == 1:
            return outputs
        # classification: logits + confidence summary stats
        probs = F.softmax(outputs, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1, keepdim=True)
        max_prob = probs.max(dim=-1, keepdim=True).values
        return torch.cat([outputs, max_prob, entropy], dim=-1)

    def _per_sample_errors(self, outputs: Tensor, labels: Tensor) -> Tensor:
        if self.task == "regression":
            preds = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
            targets = labels.squeeze(-1).float() if labels.dim() > 1 else labels.float()
            return (preds - targets).pow(2)

        logits = outputs
        if logits.dim() == 1:
            logits = torch.stack([1 - logits.sigmoid(), logits.sigmoid()], dim=-1)
        elif logits.shape[1] == 1:
            p = logits.squeeze(-1).sigmoid()
            logits = torch.stack([1 - p, p], dim=-1)
            labels = labels.long()
        return F.cross_entropy(logits, labels.long(), reduction="none")
