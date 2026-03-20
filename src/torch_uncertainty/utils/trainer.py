from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.trainer.states import (
    RunningStage,
    TrainerFn,
)

from torch_uncertainty.utils.evaluation_loop import TUEvaluationLoop


class TUTrainer(Trainer):
    def __init__(
        self,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: list[int] | str | int = "auto",
        num_nodes: int = 1,
        precision: Literal[
            "16",
            "16-mixed",
            "16-true",
            "32",
            "32-true",
            "64",
            "64-true",
            "bf16",
            "bf16-mixed",
            "bf16-true",
            "transformer-engine",
            "transformer-engine-float16",
            16,
            32,
            64,
        ]
        | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        callbacks: list[Callback] | Callback | None = None,
        fast_dev_run: int | bool = False,
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int = -1,
        min_steps: int | None = None,
        max_time: str | dict[str, int] | None = None,
        limit_train_batches: float | None = None,
        limit_val_batches: float | None = None,
        limit_test_batches: float | None = None,
        limit_predict_batches: float | None = None,
        overfit_batches: float = 0,
        val_check_interval: float | None = None,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int | None = None,
        log_every_n_steps: int | None = None,
        enable_checkpointing: bool | None = None,
        enable_progress_bar: bool | None = None,
        enable_model_summary: bool | None = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
        deterministic: bool | None | Literal["warn"] = None,
        benchmark: bool | None = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Profiler | str | None = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins=None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )
        self.test_loop = TUEvaluationLoop(
            self,
            TrainerFn.TESTING,
            RunningStage.TESTING,
            inference_mode=inference_mode,
        )
