# ruff: noqa: F401
from .aggregated_dataset import AggregatedDataset
from .classification import (
    CIFAR10C,
    CIFAR10H,
    CIFAR10N,
    CIFAR100C,
    CIFAR100N,
    CUB,
    HTRU2,
    MNISTC,
    AdultCensusIncome,
    AmazonAccess,
    APSFailure,
    BankMarketing,
    CreditApproval,
    DOTA2Games,
    GermanCredit,
    HiggsBoson,
    ImageNetA,
    ImageNetC,
    ImageNetO,
    ImageNetR,
    KDDChurn,
    NotMNIST,
    OnlineShoppers,
    OpenImageO,
    PimaDiabetes,
    SpamBase,
    TabularClassificationDataset,
    TelcoChurn,
    TinyImageNet,
    TinyImageNetC,
    UCRUEADataset,
    WineQuality,
)
from .fractals import Fractals
from .frost import FrostImages
from .kitti import KITTIDepth
from .muad import MUAD
from .nyu import NYUv2
from .regression import (
    BostonHousing,
    Concrete,
    EnergyEfficiency,
    EnergyPrediction,
    Kin8NM,
    NavalPropulsionPlant,
    PowerPlant,
    Protein,
    TabularRegressionDataset,
    Yacht,
)
from .segmentation import CamVid, Cityscapes
