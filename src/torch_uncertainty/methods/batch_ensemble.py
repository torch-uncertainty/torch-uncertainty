import torch
from einops import repeat
from torch import nn

from torch_uncertainty.layers import BatchConv2d, BatchLinear


class BatchEnsemble(nn.Module):
    def __init__(
        self,
        core_model: nn.Module,
        num_estimators: int,
        repeat_training_inputs: bool = False,
        convert_layers: bool = False,
    ) -> None:
        """Wrap a BatchEnsemble model to ensure correct batch replication.

        In a BatchEnsemble architecture, each estimator operates on a **sub-batch**
        of the input. This means that the input batch must be **repeated**
        :attr:`num_estimators` times before being processed.

        This wrapper automatically **duplicates the input batch** along the first axis,
        ensuring that each estimator receives the correct data format.

        Args:
            core_model (nn.Module): The BatchEnsemble model.
            num_estimators (int): Number of ensemble members.
            repeat_training_inputs (optional, bool): Whether to repeat the input batch during training.
                If ``True``, the input batch is repeated during both training and evaluation. If ``False``,
                the input batch is repeated only during evaluation. Default is ``False``.
            convert_layers (optional, bool): Whether to convert the model's layers to BatchEnsemble layers.
                If ``True``, the wrapper will convert all ``nn.Linear`` and ``nn.Conv2d`` layers to their
                BatchEnsemble counterparts. Default is ``False``.

        Raises:
            ValueError: If neither ``BatchLinear`` nor ``BatchConv2d`` layers are found in the core_model at the
                end of initialization.
            ValueError: If ``num_estimators`` is less than or equal to ``0``.
            ValueError: If ``convert_layers=True`` and neither ``nn.Linear`` nor ``nn.Conv2d`` layers are
                found in the core_model.

        Warning:
            If ``convert_layers==True``, the wrapper will attempt to convert all ``nn.Linear`` and ``nn.Conv2d``
            layers in the core_model to their BatchEnsemble counterparts. If the core_model contains other types of
            layers, the conversion won't happen for these layers. If don't have any ``nn.Linear`` or ``nn.Conv2d``
            layers in the core_model, the wrapper will raise an error during conversion.

        Warning:
            If ``repeat_training_inputs==True`` and you want to use one of the ``torch_uncertainty.routines``
            for training, be sure to set ``format_batch_fn=RepeatTarget(num_repeats=num_estimators)`` when
            initializing the routine.

        Example:
            >>> core_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
            >>> model = BatchEnsemble(core_model, num_estimators=4, convert_layers=True)
            >>> model
            BatchEnsemble(
            (core_model): Sequential(
                (0): BatchLinear(in_features=10, out_features=5, num_estimators=4)
                (1): ReLU()
                (2): BatchLinear(in_features=5, out_features=2, num_estimators=4)
            )
            )
        """
        super().__init__()
        self.core_model = core_model
        self.num_estimators = num_estimators
        self.repeat_training_inputs = repeat_training_inputs

        if convert_layers:
            self._convert_layers()

        filtered_modules = [
            module
            for module in self.core_model.modules()
            if isinstance(module, BatchLinear | BatchConv2d)
        ]
        _batch_ensemble_checks(filtered_modules, num_estimators)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat the input if ``self.training==False`` or ``repeat_training_inputs==True`` and pass it through the model."""
        if not self.training or self.repeat_training_inputs:
            x = repeat(x, "b ... -> (m b) ...", m=self.num_estimators)
        return self.core_model(x)

    def _convert_layers(self) -> None:
        """Convert the model's layers to BatchEnsemble layers."""
        no_valid_layers = True
        for name, layer in self.core_model.named_modules():
            if isinstance(layer, nn.Linear):
                setattr(
                    self.core_model,
                    name,
                    BatchLinear.from_linear(layer, num_estimators=self.num_estimators),
                )
                no_valid_layers = False
            elif isinstance(layer, nn.Conv2d):
                setattr(
                    self.core_model,
                    name,
                    BatchConv2d.from_conv2d(layer, num_estimators=self.num_estimators),
                )
                no_valid_layers = False
        if no_valid_layers:
            raise ValueError(
                "No valid layers found in the model. "
                "Please use `nn.Linear` or `nn.Conv2d` layers to apply BatchEnsemble."
            )


def _batch_ensemble_checks(filtered_modules: list[nn.Module], num_estimators: int) -> None:
    """Check if the model contains the required number of dropout modules."""
    if len(filtered_modules) == 0:
        raise ValueError(
            "No BatchEnsemble layers found in the model. "
            "Please use `BatchLinear` or `BatchConv2d` layers in your model "
            "or set `convert_layers=True` when initializing the wrapper."
        )
    if num_estimators <= 0:
        raise ValueError("`num_estimators` must be greater than 0.")


def batch_ensemble(
    core_model: nn.Module,
    num_estimators: int,
    repeat_training_inputs: bool = False,
    convert_layers: bool = False,
) -> BatchEnsemble:
    """BatchEnsemble wrapper for a model.

    Args:
        core_model (nn.Module): model to wrap
        num_estimators (int): number of ensemble members
        repeat_training_inputs (bool, optional): whether to repeat the input batch during training.
            If ``True``, the input batch is repeated during both training and evaluation. If ``False``,
            the input batch is repeated only during evaluation. Default is ``False``.
        convert_layers (bool, optional): whether to convert the model's layers to BatchEnsemble layers.
            If ``True``, the wrapper will convert all ``nn.Linear`` and ``nn.Conv2d`` layers to their
            BatchEnsemble counterparts. Default is ``False``.

    Returns:
        BatchEnsemble: BatchEnsemble wrapper for the :attr:`core_model`
    """
    return BatchEnsemble(
        core_model=core_model,
        num_estimators=num_estimators,
        repeat_training_inputs=repeat_training_inputs,
        convert_layers=convert_layers,
    )
