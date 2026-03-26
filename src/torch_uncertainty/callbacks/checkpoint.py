from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint, ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override


class _TUCheckpoint(Checkpoint):
    callbacks: dict[str, ModelCheckpoint]

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for callback in self.callbacks.values():
            callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in self.callbacks.values():
            callback.on_train_start(trainer=trainer, pl_module=pl_module)

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        for callback in self.callbacks.values():
            callback.on_train_batch_end(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
            )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in self.callbacks.values():
            callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in self.callbacks.values():
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

    @override
    def state_dict(self) -> dict[str, dict[str, Any]]:
        return {key: callback.state_dict() for key, callback in self.callbacks.items()}

    @override
    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        for key, callback in self.callbacks.items():
            callback.load_state_dict(state_dict=state_dict[key])

    @property
    def best_model_path(self) -> str:
        """Return the path to the best model checkpoint based on the primary metric."""
        raise NotImplementedError


class TUClsCheckpoint(_TUCheckpoint):
    def __init__(self, save_last: bool | Literal["link"] = False) -> None:
        """Keep multiple checkpoints corresponding to the best model in terms of: Accuracy,
        Expected Calibration Error, Brier-Score and Negative Log-Likelihood.

        Args:
            save_last (bool | "link", optional): When ``True``, saves a last.ckpt copy whenever a
                checkpoint file gets saved. Can be set to ``"link"`` on a local filesystem to create a
                symbolic link. This allows accessing the latest checkpoint in a deterministic
                manner. Default to ``False``.
        """
        super().__init__()
        self.callbacks = {
            "acc": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_acc={val/cls/Acc:.3f}",
                monitor="val/cls/Acc",
                mode="max",
                save_last=save_last,
                auto_insert_metric_name=False,
            ),
            "brier": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_brier={val/cls/Brier:.3f}",
                monitor="val/cls/Brier",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "nll": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_nll={val/cls/NLL:.3f}",
                monitor="val/cls/NLL",
                mode="min",
                auto_insert_metric_name=False,
            ),
        }

    @property
    def best_model_path(self) -> str:
        return self.callbacks["acc"].best_model_path


class TUSegCheckpoint(_TUCheckpoint):
    def __init__(self, save_last: bool | Literal["link"] = False) -> None:
        """Keep multiple checkpoints corresponding to the best model in terms of: Mean Intersection
        over Union, Expected Calibration Error, Brier-Score and Negative Log-Likelihood.

        Args:
            save_last (bool | "link", optional): When ``True``, saves a last.ckpt copy whenever a
                checkpoint file gets saved. Can be set to ``"link"`` on a local filesystem to create a
                symbolic link. This allows accessing the latest checkpoint in a deterministic
                manner. Default to ``False``.
        """
        super().__init__()
        self.callbacks = {
            "miou": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_miou={val/seg/mIoU:.3f}",
                monitor="val/seg/mIoU",
                mode="max",
                save_last=save_last,
                auto_insert_metric_name=False,
            ),
            "ece": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_ece={val/cal/ECE:.3f}",
                monitor="val/cal/ECE",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "brier": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_brier={val/seg/Brier:.3f}",
                monitor="val/seg/Brier",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "nll": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_nll={val/seg/NLL:.3f}",
                monitor="val/seg/NLL",
                mode="min",
                auto_insert_metric_name=False,
            ),
        }

    @property
    def best_model_path(self) -> str:
        return self.callbacks["miou"].best_model_path


class TURegCheckpoint(_TUCheckpoint):
    def __init__(
        self, probabilistic: bool = False, save_last: bool | Literal["link"] = False
    ) -> None:
        """Keep multiple checkpoints corresponding to the best model in terms of: Mean Squared
        Error, and eventually the Negative Log-Likelihood and Quantile Calibration Error.

        Args:
            probabilistic (bool, optional): If ``True``, also tracks the Negative Log-Likelihood and
                the Quantile Calibration Error. Default to ``False``.
            save_last (bool | "link", optional): When ``True``, saves a last.ckpt copy whenever a
                checkpoint file gets saved. Can be set to ``"link"`` on a local filesystem to create a
                symbolic link. This allows accessing the latest checkpoint in a deterministic
                manner. Default to ``False``.
        """
        super().__init__()
        self.callbacks = {
            "mse": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_mse={val/reg/MSE:.3f}",
                monitor="val/reg/MSE",
                mode="min",
                auto_insert_metric_name=False,
                save_last=save_last,
            ),
        }

        if probabilistic:
            self.callbacks["nll"] = ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_nll={val/reg/NLL:.3f}",
                monitor="val/reg/NLL",
                mode="min",
                auto_insert_metric_name=False,
            )
            self.callbacks["qce"] = ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_qce={val/cal/QCE:.3f}",
                monitor="val/cal/QCE",
                mode="min",
                auto_insert_metric_name=False,
            )

    @property
    def best_model_path(self) -> str:
        return self.callbacks["mse"].best_model_path
