import logging
from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_uncertainty.post_processing import PostProcessing


class Scaler(PostProcessing):
    criterion = nn.CrossEntropyLoss()
    trained = False
    logits: Tensor | None = None
    labels: Tensor | None = None

    def __init__(
        self,
        model: nn.Module | None,
        lr: float = 0.1,
        max_iter: int = 100,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Virtual class for scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate. Defaults to ``None``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to ``100``.
            eps (float): Small value for stability. Defaults to ``1e-8``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use for optimization. Defaults to ``None``.

        References:
            [1] `On calibration of modern neural networks. In ICML 2017
            <https://arxiv.org/abs/1706.04599>`_.
        """
        super().__init__(model)
        self.device = device

        if lr <= 0:
            raise ValueError(f"Learning rate must be strictly positive. Got {lr}.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError(f"Max iterations must be strictly positive. Got {max_iter}.")
        self.max_iter = int(max_iter)

        if eps <= 0:
            raise ValueError(f"Eps must be strictly positive. Got {eps}.")
        self.eps = eps

    def fit(
        self,
        dataloader: DataLoader,
        save_logits: bool = False,
        progress: bool = True,
    ) -> None:
        """Fit the temperature parameters to the calibration data.

        Args:
            dataloader (DataLoader): Dataloader with the calibration data. If there is no model,
                the dataloader should include the confidence score directly and not the logits.
            save_logits (bool, optional): Whether to save the logits and
                labels in memory. Defaults to ``False``.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to ``True``.
        """
        if self.model is None or isinstance(self.model, nn.Identity):
            logging.warning(
                "model is None. Fitting post_processing method on the dataloader's data directly."
            )
            self.model = nn.Identity()

        all_logits, all_labels = self._extract_data(dataloader, progress)
        optimizer = LBFGS(self.temperature, lr=self.lr, max_iter=self.max_iter)

        def calib_eval() -> float:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(calib_eval)
        self.trained = True
        if save_logits:
            self.logits = all_logits
            self.labels = all_labels

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if self.model is None or not self.trained:
            logging.error(
                "TemperatureScaler has not been trained yet. Returning manually tempered inputs."
            )
        return self._scale(self.model(inputs))

    def _extract_data(self, dataloader: DataLoader, progress: bool) -> tuple[Tensor, Tensor]:
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, disable=not progress):
                logits = self.model(inputs.to(self.device))
                all_logits.append(logits)
                all_labels.append(labels)
            all_logits = torch.cat(all_logits).to(self.device)
            all_labels = torch.cat(all_labels).to(self.device)

        # Handle binary classification case
        if all_logits.dim() == 2 and all_logits.shape[1] == 1:
            all_logits = all_logits.squeeze(1)
        # Stabilize optimization
        if all_logits.dim() == 1:
            all_logits = all_logits.clamp(self.eps, 1 - self.eps)
            # allow labels as probabilities
            if ((all_labels != 0) * (all_labels != 1)).sum(dtype=torch.int) != 0:
                all_labels = torch.stack([1 - all_labels, all_labels], dim=1)
            all_logits = torch.stack([torch.log(1 - all_logits), torch.log(all_logits)], dim=1)

        if all_labels.ndim == 1:
            all_labels = all_labels.long()
        return all_logits, all_labels

    @abstractmethod
    def _scale(self, logits: Tensor) -> Tensor:
        """Scale the logits with the optimal temperature.

        Args:
            logits (Tensor): Logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """

    def fit_predict(
        self,
        dataloader: DataLoader,
        progress: bool = True,
    ) -> Tensor:
        self.fit(dataloader, save_logits=True, progress=progress)
        return self(self.logits)

    @property
    @abstractmethod
    def temperature(self) -> list[Tensor]:
        """Get the temperature parameters."""
