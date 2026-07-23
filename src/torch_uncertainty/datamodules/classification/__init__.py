# ruff: noqa: F401
from .cifar10 import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule
from .imagenet import ImageNetDataModule
from .imagenet200 import ImageNet200DataModule
from .mnist import MNISTDataModule
from .tabular import (
    AdultCensusIncomeDataModule,
    AmazonAccessDataModule,
    APSFailureDataModule,
    BankMarketingDataModule,
    CreditApprovalDataModule,
    DOTA2GamesDataModule,
    GermanCreditDataModule,
    HiggsBosonDataModule,
    HTRU2DataModule,
    KDDChurnDataModule,
    OnlineShoppersDataModule,
    PimaDiabetesDataModule,
    SpamBaseDataModule,
    TabularClassificationDataModule,
    TelcoChurnDataModule,
    WineQualityDataModule,
)
from .tiny_imagenet import TinyImageNetDataModule
from .ucr_uea import UCRUEADataModule
