# ruff: noqa: F401
from .bayesian import ELBOLoss, KLDiv
from .classification import (
    BCEWithLogitsLSLoss,
    ConfidencePenaltyLoss,
    ConflictualLoss,
    CrossEntropyMaxSupLoss,
    DECLoss,
    FocalLoss,
    MixupMPLoss,
)
from .regression import BetaNLL, DERLoss, DistributionNLLLoss
