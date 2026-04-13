import copy

import torch
from torch import Tensor, nn


class CheckpointCollector(nn.Module):
    def __init__(
        self,
        core_model: nn.Module,
        cycle_start: int | None = None,
        cycle_length: int | None = None,
        save_schedule: list[int] | None = None,
        use_final_model: bool = True,
        store_on_cpu: bool = False,
    ) -> None:
        """Ensemble of models at different points in the training trajectory.

        CheckpointCollector can be used to collect samples of the posterior distribution,
        either using classical stochastic gradient optimization methods, or SGLD and SGHMC
        as implemented in TorchUncertainty.

        Args:
            core_model (nn.Module): The model to train and ensemble.
            cycle_start (int): Epoch to start ensembling. Defaults to ``None``.
            cycle_length (int): Number of epochs between model collections. Defaults to ``None``.
            save_schedule (list[int] | None): The epochs at which to save the model. Defaults to ``None``.
            use_final_model (bool): Whether to use the final model as a checkpoint. Defaults to ``True``.
            store_on_cpu (bool): Whether to put the models on the CPU when unused. Defaults to ``False``.

        Note:
            The models are saved at the end of the specified epochs.

        Note:
            If :attr:`cycle_start`, :attr:`cycle_length` and :attr:`save_schedule` are ``None``,
            the wrapper will save the models at each epoch.

        Reference:
            Checkpoint Ensembles: Ensemble Methods from a Single Training Process.
            Hugh Chen, Scott Lundberg, Su-In Lee. In ArXiv 2018.
        """
        super().__init__()
        self.mode = None
        if cycle_start is None and cycle_length is None and save_schedule is None:
            self.mode = "all"
        elif cycle_start is not None and cycle_length is not None and save_schedule is None:
            self.mode = "cycle"
        elif save_schedule is not None and cycle_start is None and cycle_length is None:
            self.mode = "schedule"
        else:
            raise ValueError(
                f"The combination of arguments: cycle_start: {cycle_start}, cycle_length: {cycle_length}, save_schedule: {save_schedule} is not known."
            )

        self.core_model = core_model
        self.cycle_start = cycle_start
        self.cycle_length = cycle_length
        self.save_schedule = save_schedule

        self.use_final_model = use_final_model
        self.store_on_cpu = store_on_cpu
        self.register_buffer("num_estimators", torch.tensor(use_final_model, dtype=torch.long))
        self.saved_models = nn.ModuleList()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        self.saved_models = nn.ModuleList()
        for _ in range(state_dict["model.num_estimators"] - 1):
            self.saved_models.append(copy.deepcopy(self.core_model))
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.saved_models = nn.ModuleList()
        for _ in range(state_dict["model.num_estimators"] - 1):
            self.saved_models.append(copy.deepcopy(self.core_model))
        return super().load_state_dict(state_dict, strict, assign)

    @torch.no_grad()
    def update_wrapper(self, epoch: int) -> None:
        """Save the model at the end of the epoch, if included in the schedule.

        Args:
            epoch (int): The current epoch.
        """
        match self.mode:
            case "schedule":
                if epoch not in self.save_schedule:
                    return
            case "cycle":
                if epoch < self.cycle_start or (epoch - self.cycle_start) % self.cycle_length != 0:
                    return
        self.saved_models.append(
            copy.deepcopy(self.core_model)
            if not self.store_on_cpu
            else copy.deepcopy(self.core_model).cpu()
        )
        self.num_estimators += 1

    def eval_forward(self, x: Tensor) -> Tensor:
        """Forward pass for evaluation.

        This method will return the ensemble prediction if models have already been collected.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The ensemble output.
        """
        preds: list[Tensor] = []
        if not len(self.saved_models):
            if self.store_on_cpu:
                preds = self.core_model.to(x.device).forward(x)
                self.core_model.cpu()
                return preds
            return self.core_model.forward(x)
        if self.store_on_cpu:
            for model in self.saved_models:
                preds.append(model.to(x.device).forward(x))
                model.cpu()
        else:
            preds = [model.forward(x) for model in self.saved_models]
        if self.use_final_model:
            preds.append(self.core_model.to(x.device).forward(x))
            if self.store_on_cpu:
                self.core_model.cpu()
        return torch.cat(preds, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for training and evaluation mode.

        If the model is in evaluation mode, this method will return the
        ensemble prediction. Otherwise, it will return the prediction of the
        current model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The model or ensemble output.
        """
        if self.training:
            if self.store_on_cpu:
                preds = self.core_model.to(x.device).forward(x)
                self.core_model.cpu()
                return preds
            return self.core_model.forward(x)
        return self.eval_forward(x)

    def to(self, *args, **kwargs):
        """Move the model and change its type.

        If :attr:`store_on_cpu` is set to True, we force device to "cpu" to avoid filling the VRAM.
        """
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)[:3]

        if self.store_on_cpu:
            device = torch.device("cpu")
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)
