Evaluating Models
=================

.. role:: bash(code)
    :language: bash

Evaluation is a first-class citizen of TorchUncertainty. Each routine wraps your model with a
rich set of metrics that go far beyond plain accuracy or MSE: proper scores, calibration,
selective prediction, ensemble diversity, OOD detection and shift robustness are all
computed automatically at validation and test time, and logged to your Lightning logger
(and optionally to a CSV file via the ``save_to_csv`` argument).

This page summarises **which metrics are computed by default** for each supported task, and
which additional metrics are enabled when you set ``eval_ood=True`` or ``eval_shift=True``
on the corresponding routine. All metrics live in
:mod:`torch_uncertainty.metrics` and most of them are documented in the
:doc:`API reference <api>`. For a list of the papers they come from, see the
:doc:`references <references>` page.

.. note::
    The metrics computed at validation time are a subset of those computed at test time —
    they are meant to monitor training and are kept lightweight. The breakdown below
    focuses on **test-time** evaluation, which is where uncertainty matters most.

----

Classification
--------------

The :class:`~torch_uncertainty.routines.ClassificationRoutine` is the most feature-complete
routine. It is suited for both binary and multi-class classification and supports
single models, ensembles, post-hoc calibration and conformal prediction.

Default test metrics
^^^^^^^^^^^^^^^^^^^^

Whatever the model, the following metrics are always computed on the in-distribution test
set:

**Performance & proper scores**

- ``cls/Acc`` — top-1 :class:`~torchmetrics.classification.Accuracy`.
- ``cls/Brier`` — :class:`~torch_uncertainty.metrics.classification.BrierScore`, a strictly
  proper score measuring the squared distance between the predicted probabilities and the
  one-hot target.
- ``cls/NLL`` — :class:`~torch_uncertainty.metrics.classification.CategoricalNLL`, the
  negative log-likelihood of the categorical predictive distribution.
- ``cls/Entropy`` — :class:`~torch_uncertainty.metrics.classification.Entropy` of the
  predictive distribution, a popular *total* uncertainty estimator.

**Calibration**

- ``cal/ECE`` — :class:`~torch_uncertainty.metrics.classification.CalibrationError`
  (Expected Calibration Error) with equal-width bins.
- ``cal/aECE`` — adaptive (equal-mass) variant of the ECE
  (:class:`~torch_uncertainty.metrics.classification.AdaptiveCalibrationError`).
- ``cal/MCE`` — Maximum Calibration Error (same metric, ``norm="max"``).
- ``cal/SmECE`` — :class:`~torch_uncertainty.metrics.classification.SmoothCalibrationError`,
  a kernel-smoothed, bin-free estimator of the calibration error.

The number of bins is controlled by ``num_bins_calibration_error`` (default ``15``). It is 
an argument of the routine itself.

**Selective Classification**

These metrics evaluate the ability of an uncertainty score to *reject* uncertain inputs
while keeping accurate predictions.

- ``sc/AURC`` — :class:`~torch_uncertainty.metrics.classification.AURC`, Area Under the
  Risk-Coverage curve.
- ``sc/AUGRC`` — :class:`~torch_uncertainty.metrics.classification.AUGRC`, Area Under the
  *Generalized* Risk-Coverage curve.
- ``sc/Cov_5Risk`` — :class:`~torch_uncertainty.metrics.classification.CovAt5Risk`, maximum
  coverage at which the selective risk stays below 5%.
- ``sc/Risk_80Cov`` — :class:`~torch_uncertainty.metrics.classification.RiskAt80Cov`,
  selective risk at 80% coverage.

**Complexity**

- ``test/cplx/flops`` and ``test/cplx/params`` are also logged for reference.

**Binary classification only**

When ``num_classes == 1``, three additional metrics are logged: ``cls/AUROC``,
``cls/AUPR`` and ``cls/FPR95`` (:class:`~torch_uncertainty.metrics.classification.FPR95`).

**Ensemble-specific metrics**

When ``is_ensemble=True`` the routine additionally computes per-sample *epistemic*
uncertainty estimators that exploit the diversity between estimators:

- ``test/ens_Disagreement`` — :class:`~torch_uncertainty.metrics.classification.Disagreement`,
- ``test/ens_MI`` — :class:`~torch_uncertainty.metrics.classification.MutualInformation`
  between the prediction and the model parameters,
- ``test/ens_Entropy`` — entropy of the **per-estimator** predictions. When ``is_ensemble=True``,
  ``test/Entropy`` corresponds to the final (average) distribution's entropy over all 
  test samples. It follows that ``test/Entropy`` = ``test/ens_MI`` + ``test/ens_Entropy``

**Post-processing & conformal prediction**

When a ``post_processing`` method (e.g., temperature scaling, Laplace) is passed, the
exact same set of test metrics is recomputed on the post-processed probabilities under
the ``test/post/`` prefix. For :class:`~torch_uncertainty.post_processing.Conformal`,
the routine instead logs ``test/post/CoverageRate``
(:class:`~torch_uncertainty.metrics.classification.CoverageRate`) and
``test/post/SetSize`` (:class:`~torch_uncertainty.metrics.classification.SetSize`). See the
:doc:`post-hoc calibration <auto_tutorials/Post_Hoc_Methods/tutorial_scalers>` and
:doc:`conformal prediction <auto_tutorials/Post_Hoc_Methods/tutorial_conformal>` tutorials
for end-to-end examples.

**Grouping Loss (optional)**

Setting ``eval_grouping_loss=True`` enables
:class:`~torch_uncertainty.metrics.classification.GroupingLoss`, a finer-grained calibration
metric that quantifies how much information is *lost* by reducing predictions to their
confidence: see Perez-Lebel et al., ICLR 2023.

OOD detection metrics
^^^^^^^^^^^^^^^^^^^^^

When ``eval_ood=True`` and the datamodule exposes an OOD dataloader, a binary
ID-vs-OOD detection task is constructed from an *OOD score* derived from the model
(controlled by the ``ood_criterion`` argument, see
:mod:`torch_uncertainty.ood_criteria` — MSP, max-logit, energy, mutual information,
post-processing-based, etc.). The following metrics are logged under the ``ood/`` prefix:

- ``ood/AUROC`` — Area Under the ROC Curve of the binary detector.
- ``ood/AUPR`` — Area Under the Precision-Recall Curve.
- ``ood/FPR95`` — :class:`~torch_uncertainty.metrics.classification.FPR95`,
  false-positive rate at 95% true-positive rate, the standard OOD-detection threshold
  metric.
- ``ood/SCOD_AURC`` — :class:`~torch_uncertainty.metrics.classification.SCODAURC`,
  SCOD Area Under the Risk-Coverage curve.
- ``ood/SCOD_AUGRC`` — :class:`~torch_uncertainty.metrics.classification.SCODAUGRC`,
  SCOD Area Under the Generalized Risk-Coverage curve.
- ``ood/SCOD_Cov_5Risk`` —
  :class:`~torch_uncertainty.metrics.classification.SCODCovAt5Risk`.
- ``ood/SCOD_Risk_80Cov`` —
  :class:`~torch_uncertainty.metrics.classification.SCODRiskAt80Cov`.
- ``ood/Entropy`` — average entropy of the predictive distribution over OOD samples.
- For ensembles, the diversity metrics above are also recomputed under the
  ``ood/ens_`` prefix.

Have a look at the :doc:`OOD detection tutorial <auto_tutorials/Classification/tutorial_ood_detection>`
for a full example.

Distribution-shift metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``eval_shift=True`` evaluates the model on a shifted version of the test set
(e.g., CIFAR-10-C, ImageNet-C). The full classification metric collection is recomputed
under the ``shift/`` prefix (``shift/cls/Acc``, ``shift/cal/ECE``, …), along with:

- ``shift/Entropy`` — average predictive entropy under shift.
- ``shift/severity`` — the corruption severity reported by the datamodule.
- For ensembles, diversity metrics under the ``shift/ens_`` prefix.

See the :doc:`distribution-shift tutorial <auto_tutorials/Classification/tutorial_distribution_shift>`
for context.

----

Segmentation
------------

The :class:`~torch_uncertainty.routines.SegmentationRoutine` reuses much of the
classification machinery, applied per pixel. Because dense per-pixel storage would be
prohibitive for the calibration and selective-classification metrics, those are computed
on a random subsample of pixels controlled by ``metric_subsampling_rate`` (default
``1e-2``).

Default test metrics
^^^^^^^^^^^^^^^^^^^^

**Segmentation performance** (computed on every pixel)

- ``seg/mIoU`` — :class:`~torch_uncertainty.metrics.segmentation.MeanIntersectionOverUnion`.
- ``seg/mAcc`` — macro-averaged pixel accuracy.
- ``seg/pixAcc`` — overall pixel accuracy.

**Proper scores, calibration, selective classification** (on subsampled pixels)

The following metrics are the per-pixel analogues of their classification counterparts and
are evaluated on a uniformly subsampled subset of pixels:

- ``seg/Brier`` (:class:`~torch_uncertainty.metrics.classification.BrierScore`),
  ``seg/NLL`` (:class:`~torch_uncertainty.metrics.classification.CategoricalNLL`)
- ``cal/ECE``, ``cal/aECE``, ``cal/MCE``
  (:class:`~torch_uncertainty.metrics.classification.CalibrationError`),
  ``cal/SmECE`` (:class:`~torch_uncertainty.metrics.classification.SmoothCalibrationError`)
- ``sc/AURC`` (:class:`~torch_uncertainty.metrics.classification.AURC`),
  ``sc/AUGRC`` (:class:`~torch_uncertainty.metrics.classification.AUGRC`),
  ``sc/Cov_5Risk`` (:class:`~torch_uncertainty.metrics.classification.CovAt5Risk`),
  ``sc/Risk_80Cov`` (:class:`~torch_uncertainty.metrics.classification.RiskAt80Cov`).

OOD detection metrics
^^^^^^^^^^^^^^^^^^^^^

When ``eval_ood=True``, the routine treats pixels whose target label is greater than or
equal to ``num_classes`` as OOD (this matches the convention used by the MUAD
datamodule). Three dense, segmentation-specific binary metrics are then computed:

- ``ood/AUROC`` —
  :class:`~torch_uncertainty.metrics.segmentation.SegmentationBinaryAUROC`.
- ``ood/AUPR`` —
  :class:`~torch_uncertainty.metrics.segmentation.SegmentationBinaryAveragePrecision`.
- ``ood/FPR95`` —
  :class:`~torch_uncertainty.metrics.segmentation.SegmentationFPR95`.
- ``ood/SCOD_AURC`` — :class:`~torch_uncertainty.metrics.classification.SCODAURC`.
- ``ood/SCOD_AUGRC`` — :class:`~torch_uncertainty.metrics.classification.SCODAUGRC`.
- ``ood/SCOD_Cov_5Risk`` —
  :class:`~torch_uncertainty.metrics.classification.SCODCovAt5Risk`.
- ``ood/SCOD_Risk_80Cov`` —
  :class:`~torch_uncertainty.metrics.classification.SCODRiskAt80Cov`.

The OOD score is again controlled by ``ood_criterion``. See the
:doc:`MUAD segmentation tutorial <auto_tutorials/Segmentation/tutorial_muad_seg>` for an
end-to-end example.

Distribution shift
^^^^^^^^^^^^^^^^^^

Distribution-shift evaluation is not implemented yet for segmentation — passing
``eval_shift=True`` will raise a :class:`NotImplementedError`. Contributions are welcome.

----

Regression
----------

The :class:`~torch_uncertainty.routines.RegressionRoutine` supports both *point-wise*
regression and *probabilistic* regression. In the latter case, the model outputs the
parameters of a PyTorch :class:`~torch.distributions.Distribution` (e.g., a Normal, a
Laplace, a NIG, …), enabling calibration evaluation on top of the usual error metrics.

Default test metrics
^^^^^^^^^^^^^^^^^^^^

**Point-wise metrics** (always computed)

- ``reg/MAE`` — :class:`~torchmetrics.regression.MeanAbsoluteError`.
- ``reg/MSE`` — :class:`~torchmetrics.regression.MeanSquaredError`.
- ``reg/RMSE`` — root mean squared error.

When the model is probabilistic, these are computed from the
:attr:`~torch_uncertainty.routines.RegressionRoutine.dist_estimate` of the predictive
distribution (``"mean"`` by default, ``"median"`` and ``"mode"`` also supported).

**Probabilistic metrics** (when ``dist_family`` is set)

- ``reg/NLL`` — :class:`~torch_uncertainty.metrics.regression.DistributionNLL`, the
  negative log-likelihood of the predictive distribution.
- ``cal/QCE`` — :class:`~torch_uncertainty.metrics.regression.QuantileCalibrationError`,
  the regression analogue of the ECE based on the empirical CDF of normalized residuals.

See the :doc:`probabilistic regression tutorial <auto_tutorials/Regression/tutorial_probabilistic_regression>`
and the :doc:`Deep Evidential Regression tutorial <auto_tutorials/Regression/tutorial_der_cubic>` for
worked examples.

OOD detection & distribution shift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OOD detection and shift evaluation are not implemented yet for regression — both will
raise a :class:`NotImplementedError` if requested. Please open an issue if you need them.

----

Pixelwise Regression
--------------------

The :class:`~torch_uncertainty.routines.PixelRegressionRoutine` is designed for dense
regression tasks (monocular depth estimation in particular) and ships with the metrics
commonly reported in the depth-estimation literature.

Default test metrics
^^^^^^^^^^^^^^^^^^^^

**Point-wise metrics**

- ``reg/SILog`` — :class:`~torch_uncertainty.metrics.regression.SILog`, scale-invariant
  log error (Eigen et al., NeurIPS 2014).
- ``reg/log10`` — :class:`~torch_uncertainty.metrics.regression.Log10` error.
- ``reg/ARE`` — mean *absolute relative* error
  (:class:`~torch_uncertainty.metrics.regression.MeanGTRelativeAbsoluteError`).
- ``reg/RSRE`` — root *squared relative* error
  (:class:`~torch_uncertainty.metrics.regression.MeanGTRelativeSquaredError`).
- ``reg/RMSE`` and ``reg/RMSELog`` — root mean squared error in linear and log space
  (:class:`~torch_uncertainty.metrics.regression.MeanSquaredLogError`).
- ``reg/iMAE`` and ``reg/iRMSE`` — inverse-depth errors
  (:class:`~torch_uncertainty.metrics.regression.MeanAbsoluteErrorInverse`,
  :class:`~torch_uncertainty.metrics.regression.MeanSquaredErrorInverse`).
- ``reg/d1``, ``reg/d2``, ``reg/d3`` —
  :class:`~torch_uncertainty.metrics.regression.ThresholdAccuracy`, the fraction of pixels
  whose prediction is within :math:`1.25^k` of the ground truth (the standard
  :math:`\delta_k` accuracies).

**Probabilistic metrics** (when ``dist_family`` is set)

- ``reg/NLL`` — :class:`~torch_uncertainty.metrics.regression.DistributionNLL` of the
  per-pixel predictive distribution. For ensembles, predictions from the different
  estimators are combined into a :class:`~torch.distributions.MixtureSameFamily`
  distribution before the NLL is computed.

OOD detection & distribution shift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like regression, OOD detection and shift evaluation are not implemented yet for pixel
regression and will raise :class:`NotImplementedError`.

----

Other metrics
-------------

A few additional metrics are implemented but **not** plugged into a routine by default;
you can use them directly on stored predictions, or wire them into a custom routine:

- :class:`~torch_uncertainty.metrics.AUSE` — Area Under the Sparsification Error curve,
  a task-agnostic metric measuring how well an uncertainty score ranks samples by their
  true error.
- :class:`~torch_uncertainty.metrics.classification.AdaptiveCalibrationError`,
  :class:`~torch_uncertainty.metrics.classification.ClasswiseCalibrationError`,
  :class:`~torch_uncertainty.metrics.classification.VariationRatio`,
  :class:`~torch_uncertainty.metrics.classification.CovAtxRisk`,
  :class:`~torch_uncertainty.metrics.classification.RiskAtxCov`,
  :class:`~torch_uncertainty.metrics.classification.FPRx` — variants of the default
  metrics with user-controlled thresholds.

See the :doc:`API reference <api>` for the full list and the
:doc:`references <references>` page for citations.
