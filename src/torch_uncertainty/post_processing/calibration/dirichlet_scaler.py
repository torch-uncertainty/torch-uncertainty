import logging
from typing import Literal

import torch
from torch import Tensor, device, nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from .matrix_scaler import MatrixScaler


class DirichletScaler(MatrixScaler):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module | None = None,
        init_w: float = 1,
        init_b: float = 0,
        lr: float = 0.1,
        max_iter: int = 200,
        lambda_reg: float | None = None,
        mu_reg: float | None = None,
        eps: float = 1e-8,
        device: Literal["cpu", "cuda"] | device | None = None,
    ) -> None:
        """Dirichlet scaling post-processing for calibrated probabilities.

        Args:
            num_classes (int): Number of classes.
            model (nn.Module | None): Model to calibrate. Defaults to ``None``.
            init_w (float, optional): Initial value for the weight matrix. Defaults to ``1``.
            init_b (float, optional): Initial value for the bias vector. Defaults to ``0``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to ``200``.
            lambda_reg (float | None, optional): Regularization coefficient applied to the
                off-diagonal elements of the weight matrix. Used to mitigate overfitting.
                Defaults to ``None``.
            mu_reg (float | None, optional): Regularization coefficient applied to the
                bias vector. Defaults to ``None``.
            eps (float): Small value for numerical stability. Defaults to ``1e-8``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use for optimization.
                Defaults to ``None``.

        References:
            [1] `Beyond temperature scaling: Obtaining well-calibrated multiclass
            probabilities with Dirichlet calibration <https://arxiv.org/abs/1910.12656>`_.
        """
        super().__init__(
            num_classes=num_classes,
            model=model,
            init_w=init_w,
            init_b=init_b,
            lr=lr,
            max_iter=max_iter,
            eps=eps,
            device=device,
        )
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg

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
            if self.lambda_reg is not None:
                loss += (
                    self.lambda_reg
                    * (self.temp_w.sum() - self.temp_w.diagonal().sum())
                    / (self.num_classes * (self.num_classes - 1))
                )
            if self.mu_reg is not None:
                loss += self.mu_reg * self.temp_b.mean()
            loss.backward()
            return loss

        optimizer.step(calib_eval)
        self.trained = True
        if save_logits:
            self.logits = all_logits
            self.labels = all_labels

    # Compute the product with the logprobs instead of the logits
    def _scale(self, logits: Tensor) -> Tensor:
        return self.temp_w @ torch.log_softmax(logits, dim=1) + self.temp_b

    @property
    def temperature(self) -> list:
        return [self.temp_w, self.temp_b]
