import pytest
import torch

from tests._dummies import dummy_model
from torch_uncertainty.methods import deep_ensembles


class TestDeepEnsemblesModel:
    """Testing the deep_ensembles function."""

    def test_main(self) -> None:
        model_1 = dummy_model(1, 10)
        model_2 = dummy_model(1, 10)

        de = deep_ensembles([model_1, model_2])
        # Check B N C
        assert de(torch.randn(3, 4, 4)).shape == (6, 10)

    def test_list_and_num_estimators(self) -> None:
        model_1 = dummy_model(1, 10)
        model_2 = dummy_model(1, 10)
        with pytest.raises(ValueError):
            deep_ensembles([model_1, model_2], num_estimators=2)

    def test_list_singleton(self) -> None:
        model_1 = dummy_model(1, 10)

        deep_ensembles([model_1], num_estimators=2, reset_model_parameters=True)
        deep_ensembles(model_1, num_estimators=2, reset_model_parameters=False)

        with pytest.raises(ValueError):
            deep_ensembles([model_1], num_estimators=1)

    def test_store_on_cpu(self) -> None:
        model_1 = dummy_model(1, 10)
        model_2 = dummy_model(1, 10)

        de = deep_ensembles([model_1, model_2], store_on_cpu=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        de.to(device)
        assert de.store_on_cpu
        assert de.core_models[0].linear.weight.device == torch.device("cpu")
        assert de.core_models[1].linear.weight.device == torch.device("cpu")

        inputs = torch.randn(3, 4, 1).to(device)
        out = de(inputs)
        assert out.device == inputs.device
        assert de.core_models[0].linear.weight.device == torch.device("cpu")
        assert de.core_models[1].linear.weight.device == torch.device("cpu")

        de = deep_ensembles([model_1, model_2], store_on_cpu=False)
        de.to(device)
        assert not de.store_on_cpu
        assert de.core_models[0].linear.weight.device == inputs.device
        assert de.core_models[1].linear.weight.device == inputs.device

    def test_error_prob_regression(self) -> None:
        # The output dicts will have different keys
        model_1 = dummy_model(1, 2, dist_family="normal")
        model_2 = dummy_model(1, 2, dist_family="nig")

        de = deep_ensembles([model_1, model_2], task="regression", probabilistic=True)

        with pytest.raises(ValueError):
            de(torch.randn(5, 1))

    def test_store_on_cpu_prob_regression(self) -> None:
        # The output dicts will have different keys
        model_1 = dummy_model(1, 2, dist_family="normal")
        model_2 = dummy_model(1, 2, dist_family="normal")

        de = deep_ensembles(
            [model_1, model_2], task="regression", probabilistic=True, store_on_cpu=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        de.to(device)
        assert de.store_on_cpu
        assert de.core_models[0].linear.weight.device == torch.device("cpu")
        assert de.core_models[1].linear.weight.device == torch.device("cpu")
        inputs = torch.randn(3, 4, 1).to(device)
        out = de(inputs)
        assert out["loc"].device == inputs.device
        assert de.core_models[0].linear.weight.device == torch.device("cpu")
        assert de.core_models[1].linear.weight.device == torch.device("cpu")

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            deep_ensembles([])

        model_1 = dummy_model(1, 10)
        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=None)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=-1)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=1)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=2, task="regression")

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=2, task="other")
