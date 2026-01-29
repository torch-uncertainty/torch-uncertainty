from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.layers.mc_batch_norm import MCBatchNorm2d
from torch_uncertainty.post_processing import PostProcessing


class MCBatchNorm(PostProcessing):
    counter: int = 0
    mc_batch_norm_layers: list[MCBatchNorm2d] = []
    trained = False

    def __init__(
        self,
        model: nn.Module | None = None,
        num_estimators: int = 16,
        convert: bool = True,
        mc_batch_size: int = 32,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Monte Carlo Batch Normalization wrapper for 2d inputs.

        Args:
            model (nn.Module): model to be converted.
            num_estimators (int): number of estimators.
            convert (bool): whether to convert the model. Defaults to ``True``.
            mc_batch_size (int, optional): Monte Carlo batch size. The smaller the more variability
                in the predictions. Defaults to ``32``.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to ``None``.

        Warning:
            The update of the batch statistics slightly differs from the method as worded in the
            original paper but sticks to its implementation. Instead of updating the training-based
            statistics with 1 new batch of data, we perform a direct replacement.
            See `this issue/discussion <https://github.com/torch-uncertainty/torch-uncertainty/issues/218>`_.

        Note:
            This wrapper will be stochastic in eval mode only.

        Note: 
            Raise an issue if you would like a wrapper for 1d and 3d inputs.

        References:
            [1] `Teye M, Azizpour H, Smith K. Bayesian uncertainty estimation for batch normalized deep networks. In ICML 2018
            <https://arxiv.org/abs/1802.06455>`_.
        """
        super().__init__()
        self.num_estimators = num_estimators
        self.convert = convert
        self.mc_batch_size = mc_batch_size
        self.device = device

        if model is not None:
            self._setup_model(model)

    def _setup_model(self, model) -> None:
        _mcbn_checks(model, self.num_estimators, self.mc_batch_size, self.convert)
        self.model = deepcopy(model)  # TODO: Is it necessary?
        self.model = self.model.eval()
        if self.convert:
            self._convert()
            if not has_mcbn(self.model):
                raise ValueError("model does not contain any MCBatchNorm2d after conversion.")

    def set_model(self, model: nn.Module) -> None:
        self.model = model
        self._setup_model(model)

    def fit(self, dataloader: DataLoader) -> None:
        """Fit the model on the dataset.

        Args:
            dataloader (DataLoader): DataLoader with the post-processing dataset.

        Warning:
            The ``batch_size`` of the DataLoader (i.e. :attr:`mc_batch_size`) should be carefully
            chosen as it will have an impact on diversity of the statistics of the MC BatchNorm
            layers and therefore of the predictions.

        Note:
            This method is used to populate the MC BatchNorm layers.
            Use the post-processing dataset.

        Raises:
            ValueError: If there are less batches than the number of estimators.
        """
        dataloader = init_dataloader(dataloader, batch_size=self.mc_batch_size)
        self.counter = 0
        self.reset_counters()
        self.set_accumulate(True)
        self.eval()
        for x, _ in dataloader:
            self.model(x.to(self.device))
            self.raise_counters()
            if self.counter == self.num_estimators:
                self.set_accumulate(False)
                self.trained = True
                return
        raise ValueError("The dataset is too small to populate the MC BatchNorm statistics.")

    def _est_forward(self, x: Tensor) -> Tensor:
        """Forward pass of a single estimator."""
        logit = self.model(x)
        self.raise_counters()
        return logit

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        if self.training:
            return self.model(inputs)
        if not self.trained:
            raise RuntimeError("MCBatchNorm has not been fit. Call .fit() first.")
        self.reset_counters()
        return torch.cat([self._est_forward(inputs) for _ in range(self.num_estimators)], dim=0)

    def _convert(self) -> None:
        """Convert all BatchNorm2d layers to MCBatchNorm2d layers."""
        self.replace_layers(self.model)

    def reset_counters(self) -> None:
        """Reset all counters to 0."""
        self.counter = 0
        for layer in self.mc_batch_norm_layers:
            layer.set_counter(0)

    def raise_counters(self) -> None:
        """Raise all counters by 1."""
        self.counter += 1
        for layer in self.mc_batch_norm_layers:
            layer.set_counter(self.counter)

    def set_accumulate(self, accumulate: bool) -> None:
        """Set the accumulate flag for all MCBatchNorm2d layers.

        Args:
            accumulate (bool): accumulate flag.
        """
        for layer in self.mc_batch_norm_layers:
            layer.accumulate = accumulate

    def replace_layers(self, model: nn.Module) -> None:
        """Replace all BatchNorm2d layers with MCBatchNorm2d layers.

        Args:
            model (nn.Module): model to be converted.
        """
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace_layers(module)

            if isinstance(module, nn.BatchNorm2d):
                mc_layer = MCBatchNorm2d(
                    num_features=module.num_features,
                    num_estimators=self.num_estimators,
                    eps=module.eps,
                    affine=module.affine,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                mc_layer.training = module.training
                mc_layer.weight = module.weight
                mc_layer.bias = module.bias
                setattr(model, name, mc_layer)

                # Save pointers to the MC BatchNorm layers
                self.mc_batch_norm_layers.append(mc_layer)


def has_mcbn(model: nn.Module) -> bool:
    """Check if the model contains any MCBatchNorm2d layers."""
    return any(isinstance(module, MCBatchNorm2d) for module in model.modules())


def init_dataloader(dataloader: DataLoader, batch_size: int):
    """Reinitialize dataloader with the chosen batch size.

    It is impossible to change the ``batch_size`` of an already-instantiated dataloader.

    Args:
        dataloader (DataLoader): the dataloader to be reinitialized with
        batch_size (int): the given batch_size.
    """
    return DataLoader(
        dataloader.dataset,
        batch_size=batch_size,
        sampler=dataloader.sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
    )


def _mcbn_checks(model, num_estimators: int, mc_batch_size: int, convert: bool) -> None:
    if num_estimators < 1 or not isinstance(num_estimators, int):
        raise ValueError(f"num_estimators must be a positive integer, got {num_estimators}.")
    if mc_batch_size < 1 or not isinstance(mc_batch_size, int):
        raise ValueError(f"mc_batch_size must be a positive integer, got {mc_batch_size}.")
    if not convert and not has_mcbn(model):
        raise ValueError("model does not contain any MCBatchNorm2d nor is not to be converted.")
