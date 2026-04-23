import copy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class SWA(nn.Module):
    num_avgd_models: Tensor

    def __init__(
        self,
        core_model: nn.Module,
        cycle_start: int,
        cycle_length: int,
    ) -> None:
        """Stochastic Weight Averaging.

        Update the SWA model every :attr:`cycle_length` epochs starting at
        :attr:`cycle_start`. Uses the SWA model only at test time. Otherwise,
        uses the base model for training.

        Args:
            core_model (nn.Module): PyTorch model to be trained.
            cycle_start (int): Epoch to start SWA.
            cycle_length (int): Number of epochs between SWA updates.

        References:
            [1] `Averaging Weights Leads to Wider Optima and Better Generalization.. In UAI 2018
            <https://arxiv.org/abs/1803.05407>`_.
        """
        super().__init__()
        _swa_checks(cycle_start, cycle_length)
        self.core_model = core_model
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length

        self.register_buffer("num_avgd_models", torch.tensor(0, device="cpu"))
        self.swa_model = None
        self.need_bn_update = False

    @torch.no_grad()
    def update_wrapper(self, epoch: int) -> None:
        if epoch >= self.cycle_start and (epoch - self.cycle_start) % self.cycle_length == 0:
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(self.core_model)
                self.num_avgd_models = torch.tensor(1)
            else:
                for swa_param, param in zip(
                    self.swa_model.parameters(),
                    self.core_model.parameters(),
                    strict=True,
                ):
                    swa_param.data += (param.data - swa_param.data) / (self.num_avgd_models + 1)
            self.num_avgd_models += 1
            self.need_bn_update = True

    def eval_forward(self, x: Tensor) -> Tensor:
        if self.swa_model is None:
            return self.core_model.forward(x)
        return self.swa_model.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)

    def bn_update(self, loader: DataLoader, device: str | torch.device | None) -> None:
        if self.need_bn_update and self.swa_model is not None:
            torch.optim.swa_utils.update_bn(loader, self.swa_model, device=device)
            self.need_bn_update = False


def _swa_checks(cycle_start: int, cycle_length: int) -> None:
    if cycle_start < 0:
        raise ValueError(f"`cycle_start` must be non-negative. Got {cycle_start}.")
    if cycle_length <= 0:
        raise ValueError(f"`cycle_length` must be strictly positive. Got {cycle_length}.")
