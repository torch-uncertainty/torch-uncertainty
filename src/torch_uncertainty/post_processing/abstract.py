from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import DataLoader


class PostProcessing(nn.Module, ABC):
    def __init__(self, model: nn.Module | None = None) -> None:
        """Abstract post-processing class."""
        super().__init__()
        self.model = model
        self.trained = False

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> None: ...

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor: ...
