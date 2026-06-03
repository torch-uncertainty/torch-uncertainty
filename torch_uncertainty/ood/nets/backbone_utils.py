from __future__ import annotations

import inspect
import logging

import numpy as np
from torch import Tensor, nn

logger = logging.getLogger(__name__)

FEATURE_OOD_CRITERIA = frozenset(
    {
        "ash",
        "scale",
        "react",
        "adascale_a",
        "vim",
        "knn",
        "neco",
        "nnguide",
    }
)

_warned_model_ids: set[int] = set()


def supports_return_feature(model: nn.Module) -> bool:
    """Whether ``forward`` accepts a ``return_feature`` keyword argument."""
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "return_feature" in sig.parameters


def get_classifier(model: nn.Module) -> nn.Module:
    """Return the final classification layer."""
    if hasattr(model, "get_fc_layer"):
        return model.get_fc_layer()
    for name in ("linear", "fc", "classification_head"):
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Module):
                return layer
    raise AttributeError(
        f"{type(model).__name__} has no classifier head "
        "(expected get_fc_layer, linear, fc, or classification_head)."
    )


def get_fc_numpy(model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """Return classifier weight and bias as NumPy arrays (OpenOOD ``get_fc`` API)."""
    if hasattr(model, "get_fc"):
        return model.get_fc()
    fc = get_classifier(model)
    if not isinstance(fc, nn.Linear):
        raise AttributeError(
            f"{type(model).__name__} classifier is not nn.Linear; cannot export get_fc weights."
        )
    w = fc.weight.detach().cpu().numpy()
    b = fc.bias.detach().cpu().numpy()
    return w, b


def get_feature_dim(model: nn.Module) -> int:
    """Penultimate feature dimension used by OOD postprocessors."""
    feature_size = getattr(model, "feature_size", None)
    if isinstance(feature_size, int):
        return feature_size
    fc = get_classifier(model)
    if isinstance(fc, nn.Linear):
        return fc.in_features
    raise AttributeError(f"{type(model).__name__} feature dimension could not be inferred.")


def warn_ood_feature_fallback(model: nn.Module) -> None:
    """Log a one-time red warning when using the ``feats_forward`` fallback."""
    key = id(model)
    if key in _warned_model_ids:
        return
    _warned_model_ids.add(key)
    red, reset = "\033[31m", "\033[0m"
    logger.warning(
        "%s%s.forward does not implement return_feature=True; using feats_forward() "
        "and the classifier head for OOD postprocessors. Implement return_feature "
        "and get_fc_layer() on the model for full OpenOOD compatibility.%s",
        red,
        type(model).__name__,
        reset,
    )


def forward_with_features(model: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(logits, features)`` using OpenOOD API or TorchUncertainty fallbacks."""
    if supports_return_feature(model):
        out = model(x, return_feature=True)
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
        raise TypeError(
            f"{type(model).__name__}.forward(return_feature=True) must return (logits, features)."
        )

    if hasattr(model, "feats_forward"):
        warn_ood_feature_fallback(model)
        feat = model.feats_forward(x)
        logits = get_classifier(model)(feat)
        return logits, feat

    raise TypeError(
        f"{type(model).__name__} cannot provide features for OOD postprocessors. "
        "Implement forward(..., return_feature=True) or feats_forward() with linear/fc."
    )


def model_supports_ood_features(model: nn.Module) -> bool:
    """Whether the model can supply features for feature-based OOD criteria."""
    if supports_return_feature(model):
        return True
    if not hasattr(model, "feats_forward"):
        return False
    try:
        get_classifier(model)
    except AttributeError:
        return False
    else:
        return True
