API Reference
=============

.. currentmodule:: torch_uncertainty

Routines
--------

Routines are the main building blocks of the library. They define the framework in which
models are trained and evaluated, and make it easy to compute the metrics crucial for
uncertainty estimation across the supported tasks: classification, regression, segmentation,
and pixelwise regression. See the :doc:`Evaluating Models <evaluation>` page for a full
breakdown of the metrics computed by each routine.

.. currentmodule:: torch_uncertainty.routines

Classification
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    ClassificationRoutine

Segmentation
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    SegmentationRoutine

Regression
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    RegressionRoutine

Pixelwise Regression
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    PixelRegressionRoutine

Layers
------

Ensemble layers
^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.layers

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    PackedLinear
    PackedConv2d
    PackedMultiheadAttention
    PackedLayerNorm
    PackedTransformerEncoderLayer
    PackedTransformerDecoderLayer
    BatchLinear
    BatchConv2d
    MaskedLinear
    MaskedConv2d


Bayesian layers
^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.layers.bayesian

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BayesLinear
    BayesConv1d
    BayesConv2d
    BayesConv3d
    LPBNNLinear
    LPBNNConv2d


Density layers
^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.layers.distributions

Linear Layers
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    NormalLinear
    LaplaceLinear
    CauchyLinear
    StudentTLinear
    NormalInverseGammaLinear

Convolution Layers
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    NormalConvNd
    LaplaceConvNd
    CauchyConvNd
    StudentTConvNd
    NormalInverseGammaConvNd

Model Backbones
---------------

.. currentmodule:: torch_uncertainty.models


ResNet
^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    batched_resnet
    lpbnn_resnet
    masked_resnet
    mimo_resnet
    packed_resnet
    resnet

WideResNet
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    batched_wideresnet28x10
    masked_wideresnet28x10
    mimo_wideresnet28x10
    packed_wideresnet28x10
    wideresnet28x10

InceptionTime
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    batched_inception_time
    bayesian_inception_time
    inception_time
    mimo_inception_time
    packed_inception_time


UQ Methods
--------------

.. currentmodule:: torch_uncertainty.methods

UQ Methods encapsulate your models to enable better uncertainty estimation.

Functions
^^^^^^^^^

Some methods can be directly created through functions such as the following: 

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    batch_ensemble
    deep_ensembles
    mc_dropout

Classes
^^^^^^^

Some methods need to be instantiated as classes such as the following:

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BatchEnsemble
    CheckpointCollector
    EMA
    StochasticModel
    SWA
    SWAG
    Zero

Metrics
-------

Classification
^^^^^^^^^^^^^^
.. currentmodule:: torch_uncertainty.metrics.classification

Proper Scores
"""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_with_ex.rst

    BrierScore

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CategoricalNLL

Out-of-Distribution Detection
"""""""""""""""""""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    FPRx
    FPR95


Selective Classification
""""""""""""""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AUGRC
    AURC
    CovAtxRisk
    CovAt5Risk
    RiskAtxCov
    RiskAt80Cov

Selective Classification with OOD
"""""""""""""""""""""""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    SCODAUGRC
    SCODAURC
    SCODCovAtxRisk
    SCODCovAt5Risk
    SCODRiskAtxCov
    SCODRiskAt80Cov

Calibration
"""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
    
    AdaptiveCalibrationError
    CalibrationError
    SmoothCalibrationError
    ClasswiseCalibrationError

Conformal Predictions
"""""""""""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
    
    CoverageRate
    SetSize

Diversity
"""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Disagreement
    Entropy
    MutualInformation
    VariationRatio


Others
""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    GroupingLoss

Regression
^^^^^^^^^^

.. currentmodule:: torch_uncertainty.metrics.regression

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    DistributionNLL
    IntervalCoverage
    IntervalScore
    Log10
    MeanAbsoluteErrorInverse
    MeanGTRelativeAbsoluteError
    MeanGTRelativeSquaredError
    MeanIntervalWidth
    MeanSquaredErrorInverse
    MeanSquaredLogError
    SILog
    ThresholdAccuracy

Calibration
"""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    QuantileCalibrationError

Segmentation
^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.metrics.segmentation

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    MeanIntersectionOverUnion
    SegmentationBinaryAUROC
    SegmentationBinaryAveragePrecision
    SegmentationFPR95

Others
^^^^^^

.. currentmodule:: torch_uncertainty.metrics

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AUSE

Losses
------

.. currentmodule:: torch_uncertainty.losses

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BCEWithLogitsLSLoss
    BetaNLL
    ConflictualLoss
    ConfidencePenaltyLoss
    DECLoss
    DERLoss
    DistributionNLLLoss
    ELBOLoss
    FocalLoss
    KLDiv
    PinballLoss

Post-Processing Methods
-----------------------

.. currentmodule:: torch_uncertainty.post_processing

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
    
    LaplaceApprox
    MCBatchNorm
    DEUP


Scaling Methods
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst

    TemperatureScaler
    VectorScaler
    MatrixScaler
    DirichletScaler
    IsotonicRegressionScaler
    PPAScaler


Conformal Methods
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst

    Conformal
    ConformalClsAPS
    ConformalClsRAPS
    ConformalClsTHR


OOD Scores
----------

.. currentmodule:: torch_uncertainty.ood_criteria

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst
    
    TUOODCriterion
    MaxLogitCriterion
    EnergyCriterion
    MaxSoftmaxCriterion
    EntropyCriterion
    MutualInformationCriterion
    PostProcessingCriterion
    VariationRatioCriterion


Datamodules
-----------

.. currentmodule:: torch_uncertainty.datamodules.abstract

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst


.. currentmodule:: torch_uncertainty.datamodules

Classification
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CIFAR10DataModule
    CIFAR100DataModule
    ImageNetDataModule
    MNISTDataModule
    TinyImageNetDataModule
    

Tabular Classification
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    TabularClassificationDataModule
    AdultCensusIncomeDataModule
    AmazonAccessDataModule
    APSFailureDataModule
    BankMarketingDataModule
    CreditApprovalDataModule
    DOTA2GamesDataModule
    GermanCreditDataModule
    HiggsBosonDataModule
    HTRU2DataModule
    KDDChurnDataModule
    OnlineShoppersDataModule
    PimaDiabetesDataModule
    SpamBaseDataModule
    TelcoChurnDataModule
    WineQualityDataModule
    
Regression
^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    TabularRegressionDataModule

.. currentmodule:: torch_uncertainty.datamodules.segmentation

Segmentation
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CamVidDataModule
    CityscapesDataModule
    MUADDataModule

Datasets
--------

.. currentmodule:: torch_uncertainty.datasets

Classification
^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.classification

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    MNISTC
    NotMNIST
    CIFAR10C
    CIFAR100C
    CIFAR10H
    CIFAR10N
    CIFAR100N
    ImageNetA
    ImageNetC
    ImageNetO
    ImageNetR
    TinyImageNet
    TinyImageNetC
    OpenImageO

Tabular Classification
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.classification.tabular

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    TabularClassificationDataset
    AdultCensusIncome
    AmazonAccess
    APSFailure
    BankMarketing
    CreditApproval
    DOTA2Games
    GermanCredit
    HiggsBoson
    HTRU2
    KDDChurn
    OnlineShoppers
    PimaDiabetes
    SpamBase
    TelcoChurn
    WineQuality

Regression
^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.regression

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    TabularRegressionDataset

Segmentation
^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.segmentation

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CamVid
    Cityscapes

Others & Cross-Categories
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Fractals
    FrostImages
    KITTIDepth
    MUAD
    NYUv2

Callbacks
---------

Custom Lightning callbacks for advanced checkpointing and model saving functionalities.

.. currentmodule:: torch_uncertainty.callbacks

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CompoundCheckpoint
    TUClsCheckpoint
    TURegCheckpoint
    TUSegCheckpoint
