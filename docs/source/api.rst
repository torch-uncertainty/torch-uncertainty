API Reference
=============

.. currentmodule:: torch_uncertainty

Routines
--------

The routine are the main building blocks of the library. They define the framework
in which the models are trained and evaluated. They allow for easy computation of different
metrics crucial for uncertainty estimation in different contexts, namely classification, regression and segmentation.

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
"""""""""""""

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
""""""""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    NormalConvNd
    LaplaceConvNd
    CauchyConvNd
    StudentTConvNd
    NormalInverseGammaConvNd

Models
------

.. currentmodule:: torch_uncertainty.models

Backbones
^^^^^^^^^

ResNet
""""""
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
""""""""""

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
"""""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    batched_inception_time
    bayesian_inception_time
    inception_time
    mimo_inception_time
    packed_inception_time


Wrappers
^^^^^^^^

Functions
"""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    batch_ensemble
    deep_ensembles
    mc_dropout

Classes
"""""""

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

Calibration
"""""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
    
    AdaptiveCalibrationError
    CalibrationError

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
    Log10
    MeanAbsoluteErrorInverse
    MeanGTRelativeAbsoluteError
    MeanGTRelativeSquaredError
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

Post-Processing Methods
-----------------------

.. currentmodule:: torch_uncertainty.post_processing

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
    
    LaplaceApprox
    MCBatchNorm


Scaling Methods
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst

    MatrixScaler
    TemperatureScaler
    VectorScaler


Conformal Methods
^^^^^^^^^^^^^^^

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
    

UCI Tabular Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BankMarketingDataModule
    DOTA2GamesDataModule
    HTRU2DataModule
    OnlineShoppersDataModule
    SpamBaseDataModule
    
Regression
^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    UCIRegressionDataModule

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


UCI Tabular Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.classification.uci


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BankMarketing
    DOTA2Games
    HTRU2
    OnlineShoppers
    SpamBase


Regression
^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.regression

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    UCIRegression

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
