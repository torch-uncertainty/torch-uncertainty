# ruff: noqa: F401
from .batch_ensemble import BatchEnsemble, batch_ensemble
from .checkpoint_collector import (
    CheckpointCollector,
)
from .deep_ensembles import deep_ensembles
from .ema import EMA
from .mc_dropout import mc_dropout
from .stochastic import StochasticModel
from .swa import SWA
from .swag import SWAG
from .zero import Zero

STEP_UPDATE_MODEL = (EMA,)
EPOCH_UPDATE_MODEL = (SWA, SWAG, CheckpointCollector)
