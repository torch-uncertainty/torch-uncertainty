from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import DataLoader


class PostProcessing(nn.Module, ABC):
    def __init__(self, model: nn.Module | None = None) -> None:
        """Abstract base class for post-processing modules.

        Post-processing modules wrap a trained model and transform its outputs at test
        time — for example to calibrate its probabilities (temperature, vector, matrix
        scaling, histogram binning, isotonic regression, etc.) or to turn them into
        prediction sets (conformal prediction).

        Subclasses must implement :meth:`fit`, which is called once on a held-out
        calibration dataloader before evaluation, and :meth:`forward`, which applies
        the post-processing to the model outputs at test time.

        Args:
            model: The model to wrap. Can be set later via :meth:`set_model`. Defaults to ``None``.

        Attributes:
            model: The wrapped model.
            trained: Whether :meth:`fit` has already been called. Subclasses should set
                this to ``True`` once fitting is done.
        """
        super().__init__()
        self.model = model
        self.trained = False

    def set_model(self, model: nn.Module) -> None:
        """Attach a model to the post-processing module."""
        self.model = model

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> None:
        """Fit the post-processing module on a calibration dataloader.

        Args:
            dataloader: A dataloader yielding ``(inputs, targets)`` pairs from a
                held-out calibration set, disjoint from both the training and the
                test sets.
        """

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the post-processing transform to a batch of inputs.

        Args:
            inputs: A batch of inputs to feed to the wrapped model.

        Returns:
            The post-processed model outputs (typically calibrated probabilities or
            prediction sets, depending on the subclass).
        """
