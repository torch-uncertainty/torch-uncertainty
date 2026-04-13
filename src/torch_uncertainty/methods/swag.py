import copy
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .swa import SWA


class SWAG(SWA):
    swag_stats: dict[str, Tensor]
    prfx = "model.swag_stats."

    def __init__(
        self,
        core_model: nn.Module,
        cycle_start: int,
        cycle_length: int,
        scale: float = 1.0,
        diag_covariance: bool = False,
        max_num_models: int = 20,
        var_clamp: float = 1e-6,
        num_estimators: int = 16,
    ) -> None:
        """Stochastic Weight Averaging Gaussian (SWAG).

        Update the SWAG posterior every `cycle_length` epochs starting at
        `cycle_start`. Samples :attr:`num_estimators` models from the SWAG
        posterior after each update. Uses the SWAG posterior estimation only
        at test time. Otherwise, uses the base model for training.

        Call :meth:`update_wrapper` at the end of each epoch. It will update
        the SWAG posterior if the current epoch number minus :attr:`cycle_start`
        is a multiple of :attr:`cycle_length`. Call :meth:`bn_update` to update
        the batchnorm statistics of the current SWAG samples.

        Args:
            core_model (nn.Module): PyTorch model to be trained.
            cycle_start (int): Begininning of the first SWAG averaging cycle.
            cycle_length (int): Number of epochs between SWAG updates. The first update occurs at :attr:`cycle_start` + :attr:`cycle_length`.
            scale (float, optional): Scale of the Gaussian. Defaults to ``1.0``.
            diag_covariance (bool, optional): Whether to use a diagonal covariance. Defaults to ``False``.
            max_num_models (int, optional): Maximum number of models to store. Defaults to ``0``.
            var_clamp (float, optional): Minimum variance. Defaults to ``1e-30``.
            num_estimators (int, optional): Number of posterior estimates to use. Defaults to ``16``.

        References:
            [1] `A simple baseline for bayesian uncertainty in deep learning. In NeurIPS 2019
            <https://arxiv.org/abs/1902.02476>`_.

        Note:
            Modified from https://github.com/wjmaddox/swa_gaussian.
        """
        super().__init__(core_model, cycle_start, cycle_length)
        _swag_checks(scale, max_num_models, var_clamp)

        self.num_estimators = num_estimators
        self.scale = scale

        self.diag_covariance = diag_covariance
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        self.initialize_stats()
        self.fit = False
        self.samples = []

    def eval_forward(self, x: Tensor) -> Tensor:
        """Forward pass of the SWAG model when in eval mode."""
        if not self.fit:
            return self.core_model.forward(x)
        return torch.cat([mod.to(device=x.device)(x) for mod in self.samples])

    def initialize_stats(self) -> None:
        """Initialize the SWAG dictionary of statistics.

        For each parameter, we create a mean, squared mean, and covariance
        square root. The covariance square root is only used when
        `diag_covariance` is False.
        """
        self.swag_stats = {}
        for name_p, param in self.core_model.named_parameters():
            mean, squared_mean = (
                torch.zeros_like(param, device="cpu"),
                torch.zeros_like(param, device="cpu"),
            )
            self.swag_stats[self.prfx + name_p + "_mean"] = mean
            self.swag_stats[self.prfx + name_p + "_sq_mean"] = squared_mean

            if not self.diag_covariance:
                covariance_sqrt = torch.zeros((0, param.numel()), device="cpu")
                self.swag_stats[self.prfx + name_p + "_covariance_sqrt"] = covariance_sqrt

    @torch.no_grad()
    def update_wrapper(self, epoch: int) -> None:
        """Update the SWAG posterior.

        The update is performed if the epoch is greater than the cycle start
        and the difference between the epoch and the cycle start is a multiple
        of the cycle length.

        Args:
            epoch (int): Current epoch.
        """
        if not (epoch > self.cycle_start and (epoch - self.cycle_start) % self.cycle_length == 0):
            return

        for name_p, param in self.core_model.named_parameters():
            mean = self.swag_stats[self.prfx + name_p + "_mean"]
            squared_mean = self.swag_stats[self.prfx + name_p + "_sq_mean"]
            new_param = param.data.detach().cpu()

            mean = mean * self.num_avgd_models / (self.num_avgd_models + 1) + new_param / (
                self.num_avgd_models + 1
            )
            squared_mean = squared_mean * self.num_avgd_models / (
                self.num_avgd_models + 1
            ) + new_param**2 / (self.num_avgd_models + 1)

            self.swag_stats[self.prfx + name_p + "_mean"] = mean
            self.swag_stats[self.prfx + name_p + "_sq_mean"] = squared_mean

            if not self.diag_covariance:
                covariance_sqrt = self.swag_stats[self.prfx + name_p + "_covariance_sqrt"]
                dev = (new_param - mean).view(-1, 1).t()
                covariance_sqrt = torch.cat((covariance_sqrt, dev), dim=0)
                if self.num_avgd_models + 1 > self.max_num_models:
                    covariance_sqrt = covariance_sqrt[1:, :]
                self.swag_stats[self.prfx + name_p + "_covariance_sqrt"] = covariance_sqrt

        self.num_avgd_models += 1

        self.samples = [
            self.sample(self.scale, self.diag_covariance) for _ in range(self.num_estimators)
        ]
        self.need_bn_update = True
        self.fit = True

    def bn_update(self, loader: DataLoader, device: torch.device) -> None:
        """Update the bachnorm statistics of the current SWAG samples.

        Args:
            loader (DataLoader): DataLoader to update the batchnorm statistics.
            device (torch.device): Device to perform the update.
        """
        if self.need_bn_update:
            for mod in self.samples:
                torch.optim.swa_utils.update_bn(loader, mod, device=device)
            self.need_bn_update = False

    def sample(
        self,
        scale: float,
        diag_covariance: bool | None = None,
        block: bool = False,
        seed: int | None = None,
    ) -> nn.Module:
        """Sample a model from the SWAG posterior.

        Args:
            scale (float): Rescale coefficient of the Gaussian.
            diag_covariance (bool, optional): Whether to use a diagonal
                covariance. Defaults to None.
            block (bool, optional): Whether to sample a block diagonal
                covariance. Defaults to False.
            seed (int, optional): Random seed. Defaults to None.

        Returns:
            nn.Module: Sampled model.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if diag_covariance is None:
            diag_covariance = self.diag_covariance
        if not diag_covariance and self.diag_covariance:
            raise ValueError("Cannot sample full rank from diagonal covariance matrix.")

        if not block:
            return self._fullrank_sample(scale, diag_covariance)
        raise NotImplementedError("Raise an issue if you need this feature.")

    def _fullrank_sample(self, scale: float, diagonal_covariance: bool) -> nn.Module:
        new_sample = copy.deepcopy(self.core_model)

        for name_p, param in new_sample.named_parameters():
            mean = self.swag_stats[self.prfx + name_p + "_mean"]
            sq_mean = self.swag_stats[self.prfx + name_p + "_sq_mean"]

            if not diagonal_covariance:
                cov_mat_sqrt = self.swag_stats[self.prfx + name_p + "_covariance_sqrt"]

            var = torch.clamp(sq_mean - mean**2, self.var_clamp)
            var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

            if not diagonal_covariance:
                cov_sample = cov_mat_sqrt.t() @ torch.randn((cov_mat_sqrt.size(0),))
                cov_sample /= (self.max_num_models - 1) ** 0.5
                var_sample += cov_sample.view_as(var_sample)

            sample = mean + scale**0.5 * var_sample
            param.data = sample.to(device="cpu", dtype=param.dtype)
        return new_sample

    def _save_to_state_dict(self, destination: dict | None, prefix: str, keep_vars: bool) -> None:
        """Add the SWAG statistics to the destination dict."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination |= self.swag_stats

    def state_dict(
        self, *args, destination: dict | None = None, prefix: str = "", keep_vars: bool = False
    ) -> Mapping:
        """Add the SWAG statistics to the state dict."""
        return self.swag_stats | super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def _load_swag_stats(self, state_dict: Mapping) -> None:
        """Load the SWAG statistics from the state dict."""
        self.swag_stats = {k: v for k, v in state_dict.items() if k in self.swag_stats}
        for k in self.swag_stats:
            del state_dict[k]
        self.samples = [
            self.sample(self.scale, self.diag_covariance) for _ in range(self.num_estimators)
        ]
        self.need_bn_update = True
        self.fit = True

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        self._load_swag_stats(state_dict)
        return super().load_state_dict(state_dict, strict, assign)

    def compute_logdet(self, block: bool = False) -> None:
        raise NotImplementedError("Raise an issue if you need this feature.")

    def compute_logprob(self, vec=None, block: bool = False, diag: bool = False) -> None:
        raise NotImplementedError("Raise an issue if you need this feature.")


def _swag_checks(scale: float, max_num_models: int, var_clamp: float) -> None:
    if scale < 0:
        raise ValueError(f"`scale` must be non-negative. Got {scale}.")
    if max_num_models < 0:
        raise ValueError(f"`max_num_models` must be non-negative. Got {max_num_models}.")
    if var_clamp < 0:
        raise ValueError(f"`var_clamp` must be non-negative. Got {var_clamp}.")
