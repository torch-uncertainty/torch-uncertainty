"""Tests for DEUP post-processing."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.model import dummy_model
from torch_uncertainty.ood_criteria import DEUPCriterion
from torch_uncertainty.post_processing import DEUP
from torch_uncertainty.post_processing.deup import _ErrorPredictor


class TestDEUP:
    """Testing the DEUP post-processing class."""

    def test_classification_fit_forward(self) -> None:
        torch.manual_seed(0)
        n, in_dim, n_classes = 80, 4, 3
        x = torch.randn(n, in_dim)
        y = torch.randint(0, n_classes, (n,))
        dl = DataLoader(TensorDataset(x, y), batch_size=16)
        model = dummy_model(in_dim, n_classes)

        deup = DEUP(task="classification", model=model, num_folds=4, max_epochs=5, device="cpu")
        deup.fit(dl)
        unc = deup(x[:8])
        assert unc.shape == (8,)
        assert torch.all(unc >= 0)

        probs = deup.predict_proba(x[:8])
        assert probs.shape == (8, n_classes)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_regression_fit_forward(self) -> None:
        torch.manual_seed(1)
        n, in_dim = 60, 3
        x = torch.randn(n, in_dim)
        y = torch.randn(n, 1)
        dl = DataLoader(TensorDataset(x, y), batch_size=12)
        model = dummy_model(in_dim, 1)

        deup = DEUP(task="regression", model=model, num_folds=3, max_epochs=5, device="cpu")
        deup.fit(dl)
        unc = deup(x[:5])
        assert unc.shape == (5,)
        assert torch.all(unc >= 0)

    def test_deup_criterion(self) -> None:
        crit = DEUPCriterion()
        scores = torch.tensor([0.1, 2.0, 0.5])
        assert torch.allclose(crit(scores), scores)

    def test_epistemic_ranks_errors_classification(self) -> None:
        """DEUP assigns higher uncertainty to regions where the model has higher CE.

        We create two clearly separated groups:
        - Confident group: large positive ``inputs[:, 0]``  → model predicts class 0
          with high probability → labels are class 0 → very low CE (~0.03).
        - Uncertain group: near-zero ``inputs[:, 0]`` → uniform logits → labels are
          wrong classes → CE ≈ log(4) ≈ 1.39.

        The features ``[logits, max_prob, entropy]`` cleanly separate the groups, so
        the error predictor should learn to rank them and DEUP scores should be
        systematically higher for the uncertain group.
        """
        torch.manual_seed(2)
        n, in_dim, n_classes = 100, 4, 4
        half = n // 2

        x = torch.randn(n, in_dim)
        x[:half, 0] = x[:half, 0].abs() + 3.0  # confident: first feature >> 0
        x[half:, 0] = x[half:, 0] * 0.05  # uncertain: first feature ≈ 0

        y = torch.zeros(n, dtype=torch.long)
        y[half:] = torch.randint(1, n_classes, (half,))  # uncertain group: wrong label

        class ConfidenceFromFirstFeature(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros(inputs.shape[0], n_classes)
                logits[:, 0] = inputs[:, 0].clamp(min=0)
                return logits

        model = ConfidenceFromFirstFeature()
        dl = DataLoader(TensorDataset(x, y), batch_size=20)
        deup = DEUP(task="classification", model=model, num_folds=4, max_epochs=50, device="cpu")
        deup.fit(dl)

        with torch.no_grad():
            unc = deup(x)

        assert unc[half:].mean() > unc[:half].mean()

    def test_invalid_init_args(self) -> None:
        with pytest.raises(ValueError, match="task must be"):
            DEUP(task="not_a_task")
        with pytest.raises(ValueError, match="num_folds must be >= 2"):
            DEUP(task="classification", num_folds=1)
        with pytest.raises(ValueError, match="hidden_dim must be >= 1"):
            DEUP(task="classification", hidden_dim=0)

    def test_runtime_errors_without_model(self) -> None:
        deup = DEUP(task="classification", model=None)
        dl = DataLoader(TensorDataset(torch.randn(4, 3), torch.zeros(4, dtype=torch.long)))
        with pytest.raises(RuntimeError, match=r"Model must be set before calling fit"):
            deup.fit(dl)
        with pytest.raises(RuntimeError, match=r"DEUP must be fitted before forward"):
            deup(torch.randn(2, 3))
        with pytest.raises(RuntimeError, match=r"Model must be set."):
            deup.predict_proba(torch.randn(2, 3))

    def test_predict_proba_requires_classification(self) -> None:
        deup = DEUP(task="regression", model=dummy_model(3, 1))
        with pytest.raises(RuntimeError, match="only defined for classification"):
            deup.predict_proba(torch.randn(2, 3))

    def test_forward_warns_when_not_trained(self, caplog) -> None:
        n_classes = 3
        deup = DEUP(task="classification", model=dummy_model(4, n_classes))
        # Features are [logits, max_prob, entropy] -> n_classes + 2 dims.
        deup.error_predictor = _ErrorPredictor(n_classes + 2, deup.hidden_dim)
        assert not deup.trained
        deup(torch.randn(2, 4))
        assert "DEUP has not been fitted" in caplog.text

    def test_set_model_resets_state(self) -> None:
        deup = DEUP(task="classification", model=dummy_model(4, 3))
        deup.trained = True
        deup.error_predictor = _ErrorPredictor(5, deup.hidden_dim)
        deup.set_model(dummy_model(4, 3))
        assert not deup.trained
        assert deup.error_predictor is None

    def test_binary_classification_fit_forward(self) -> None:
        """Exercise the single-logit (N, 1) classification paths.

        Covers the ``shape[1] == 1`` branches in feature extraction, per-sample
        error computation and ``predict_proba``.
        """
        torch.manual_seed(3)
        n, in_dim = 40, 3
        x = torch.randn(n, in_dim)
        y = torch.randint(0, 2, (n,))
        dl = DataLoader(TensorDataset(x, y), batch_size=10)
        model = dummy_model(in_dim, 1)

        deup = DEUP(task="classification", model=model, num_folds=2, max_epochs=3, device="cpu")
        deup.fit(dl)

        unc = deup(x[:5])
        assert unc.shape == (5,)
        assert torch.all(unc >= 0)

        probs = deup.predict_proba(x[:5])
        assert probs.shape == (5, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_one_dimensional_output_paths(self) -> None:
        """Exercise the ``outputs.dim() == 1`` branches.

        Covers the 1-D paths in feature extraction, per-sample error computation
        and ``predict_proba`` for a model emitting a flat logit vector.
        """
        torch.manual_seed(4)
        n, in_dim = 30, 3
        x = torch.randn(n, in_dim)
        y = torch.randint(0, 2, (n,))

        class FlatLogitModel(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs[:, 0]  # 1-D output

        deup = DEUP(
            task="classification", model=FlatLogitModel(), num_folds=2, max_epochs=3, device="cpu"
        )
        deup.fit(DataLoader(TensorDataset(x, y), batch_size=10))

        unc = deup(x[:5])
        assert unc.shape == (5,)
        assert torch.all(unc >= 0)

        probs = deup.predict_proba(x[:5])
        assert probs.shape == (5, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_regression_one_dimensional_output(self) -> None:
        """Exercise the 1-D squared-error path in regression."""
        torch.manual_seed(5)
        n, in_dim = 30, 3
        x = torch.randn(n, in_dim)
        y = torch.randn(n)

        class FlatRegModel(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs[:, 0]  # 1-D output

        deup = DEUP(
            task="regression", model=FlatRegModel(), num_folds=2, max_epochs=3, device="cpu"
        )
        deup.fit(DataLoader(TensorDataset(x, y), batch_size=10))

        unc = deup(x[:5])
        assert unc.shape == (5,)
        assert torch.all(unc >= 0)
