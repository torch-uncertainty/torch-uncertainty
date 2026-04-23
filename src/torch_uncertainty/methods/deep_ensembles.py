import copy
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn


class _DeepEnsembles(nn.Module):
    def __init__(
        self,
        core_models: list[nn.Module],
        store_on_cpu: bool = False,
    ) -> None:
        """Create a classification deep ensembles from a list of models."""
        super().__init__()
        self.core_models = nn.ModuleList(core_models)
        self.num_estimators = len(core_models)
        self.store_on_cpu = store_on_cpu

    def forward(self, x: Tensor) -> Tensor:
        r"""Return the logits of the ensemble.

        Args:
            x (Tensor): The input of the model.

        Returns:
            Tensor: The output of the model with shape :math:`(N \times B, C)`,
                where :math:`B` is the batch size, :math:`N` is the number of
                estimators, and :math:`C` is the number of classes.
        """
        preds: list[Tensor] = []
        if self.store_on_cpu:
            for model in self.core_models:
                model.to(x.device)
                preds.append(model.forward(x))
                model.cpu()
        else:
            preds = [model.forward(x) for model in self.core_models]
        return torch.cat(preds, dim=0)

    def to(self, *args, **kwargs: dict):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)[:3]

        if self.store_on_cpu:
            device = torch.device("cpu")

        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)


class _RegDeepEnsembles(_DeepEnsembles):
    def __init__(
        self,
        probabilistic: bool,
        core_models: list[nn.Module],
        store_on_cpu: bool = False,
    ) -> None:
        """Create a regression deep ensembles from a list of models."""
        super().__init__(core_models=core_models, store_on_cpu=store_on_cpu)
        self.probabilistic = probabilistic

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        r"""Return the logits of the ensemble.

        Args:
            x (Tensor): The input of the model.

        Returns:
            Tensor | dict[str, Tensor]: The output of the model with shape :math:`(N \times B, *)`
                where :math:`B` is the batch size, :math:`N` is the number of estimators, and
                :math:`*` is any other dimension.
        """
        if self.probabilistic:
            if self.store_on_cpu:
                out = []
                for model in self.core_models:
                    model.to(x.device)
                    out.append(model.forward(x))
                    model.to("cpu")
            else:
                out = [model.forward(x) for model in self.core_models]
            key_set = {tuple(o.keys()) for o in out}
            if len(key_set) != 1:
                raise ValueError("The output of the models must have the same keys.")
            return {k: torch.cat([o[k] for o in out], dim=0) for k in key_set.pop()}
        return super().forward(x)


def deep_ensembles(
    core_models: list[nn.Module] | nn.Module,
    num_estimators: int | None = None,
    task: Literal[
        "classification", "regression", "segmentation", "pixel_regression"
    ] = "classification",
    probabilistic: bool | None = None,
    reset_model_parameters: bool = True,
    store_on_cpu: bool = False,
    ckpt_paths: list[str | Path] | Path | None = None,
    use_tu_ckpt_format: bool = True,
) -> _DeepEnsembles:
    """Build a Deep Ensembles out of the original models.

    Args:
        core_models (list[nn.Module] | nn.Module): The model to be ensembled.
        num_estimators (int | None): The number of estimators in the ensemble.
        task (Literal[``"classification"``, ``"regression"``, ``"segmentation"``, ``"pixel_regression"``]): The model task. Defaults to ``"classification"``.
        probabilistic (bool): Whether the regression model is probabilistic.
        reset_model_parameters (bool): Whether to reset the model parameters
            when :attr:core_models is a module or a list of length 1. Defaults to ``True``.
        store_on_cpu (bool): Whether to store the models on CPU. Defaults to ``False``.
            This is useful for large models that do not fit in GPU memory. Only one
            model will be stored on GPU at a time during forward. The rest will be stored on CPU.
        ckpt_paths (list[str | Path] | None): The paths to the checkpoints of the models.
            If provided, the models will be loaded from the checkpoints. The number of
            models and the number of checkpoint paths must be the same. If not provided,
            the models will be used as is. Defaults to ``None``.
        use_tu_ckpt_format (bool): Whether the checkpoint is from torch-uncertainty. If ``True``,
            the checkpoint will be loaded using the torch-uncertainty loading function. If
            ``False``, the checkpoint will be loaded using the default PyTorch loading function.
            Note that this option is only used if :attr:ckpt_paths is provided. Defaults to
            ``True``.

    Returns:
        _DeepEnsembles | _RegDeepEnsembles: The ensembled model.

    Raises:
        ValueError: If :attr:`num_estimators` is not specified and :attr:`models`
            is a module (or singleton list).
        ValueError: If :attr:`num_estimators` is less than 2 and :attr:`models` is
            a module (or singleton list).
        ValueError: If :attr:`num_estimators` is defined while :attr:`models` is
            a (non-singleton) list.

    Warning:
        The :attr:`store_on_cpu` option is not supported for training. It is
        only supported for inference.

    References:
            [1] `Simple and scalable predictive uncertainty estimation using deep ensembles. In NeurIPS, 2017
            <https://arxiv.org/abs/1612.01474>`_.

    """
    if isinstance(core_models, list) and len(core_models) == 0:
        raise ValueError("Models must not be an empty list.")
    if (isinstance(core_models, list) and len(core_models) == 1) or isinstance(
        core_models, nn.Module
    ):
        if num_estimators is None:
            raise ValueError("if models is a module, num_estimators must be specified.")
        if num_estimators < 2:
            raise ValueError(f"num_estimators must be at least 2. Got {num_estimators}.")

        if isinstance(core_models, list):
            core_models = core_models[0]

        core_models = [copy.deepcopy(core_models) for _ in range(num_estimators)]

        if reset_model_parameters:
            for model in core_models:
                for layer in model.modules():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

    elif isinstance(core_models, list) and len(core_models) > 1 and num_estimators is not None:
        raise ValueError("num_estimators must be None if you provided a non-singleton list.")

    if ckpt_paths is not None:  # coverage: ignore
        if isinstance(ckpt_paths, str | Path):
            ckpt_dir = Path(ckpt_paths)
            if not ckpt_dir.is_dir():
                raise ValueError("ckpt_paths must be a directory or a list of paths.")
            ckpt_paths = sorted(
                elt
                for elt in ckpt_dir.iterdir()
                if elt.is_file() and elt.suffix in [".pt", ".pth", ".ckpt"]
            )
            if len(ckpt_paths) == 0:
                raise ValueError("No checkpoint files found in the directory.")

        if len(core_models) != len(ckpt_paths):
            raise ValueError(
                "The number of models and the number of checkpoint paths must be the same."
            )
        for model, ckpt_path in zip(core_models, ckpt_paths, strict=True):
            if isinstance(ckpt_path, str | Path):
                loaded_data = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in loaded_data:
                    state_dict = loaded_data["state_dict"]
                else:
                    state_dict = loaded_data

                if use_tu_ckpt_format:
                    model.load_state_dict(
                        {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
                    )
                else:
                    model.load_state_dict(state_dict)
            else:
                raise TypeError("Checkpoint paths must be strings or Path objects.")

    match task:
        case "classification" | "segmentation":
            return _DeepEnsembles(core_models=core_models, store_on_cpu=store_on_cpu)
        case "regression" | "pixel_regression":
            if probabilistic is None:
                raise ValueError("probabilistic must be specified for regression models.")
            return _RegDeepEnsembles(
                probabilistic=probabilistic, core_models=core_models, store_on_cpu=store_on_cpu
            )
        case _:
            raise ValueError(f"Unknown task: {task}.")
