from torch import Tensor

from .risk_coverage import AUGRC, AURC, CovAtxRisk, RiskAtxCov


class SCODAURC(AURC):
    r"""Area Under the SCOD Risk-Coverage curve.

    This metric extends selective classification to a binary
    in-distribution-vs-out-of-distribution (ID/OOD) decision problem.

    Let :math:`s_i` be an OOD score (higher means *more OOD*) and
    :math:`e_i \in \{0,1\}` the SCOD error indicator with
    :math:`e_i = 1` for OOD samples and :math:`e_i = 0` for ID samples.
    Samples are sorted by decreasing ID acceptance confidence
    :math:`-s_i`, and the risk at coverage :math:`\kappa = k/N` is:

    .. math::

        r\!\left(\tfrac{k}{N}\right) = \frac{1}{k}\sum_{i=1}^{k} e_{\sigma(i)}.

    The SCOD-AURC is then:

    .. math::

        \text{SCOD-AURC} = \int_0^1 r(\kappa)\,\mathrm{d}\kappa
        \approx \frac{1}{N}\sum_{k=1}^{N} r\!\left(\tfrac{k}{N}\right).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - **ood_scores** (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
      containing OOD scores.
    - **targets** (:class:`~torch.Tensor`): A binary int tensor of shape ``(N, ...)``
      with ``0`` for ID and ``1`` for OOD.
    """

    def update(self, ood_scores: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store SCOD confidence scores and associated detection errors.

        Args:
            ood_scores: OOD scores where higher means more OOD-like.
            targets: Binary labels, with 0 for ID and 1 for OOD.
        """
        self.scores.append(-ood_scores.reshape(-1))
        self.errors.append(targets.reshape(-1).float())


class SCODAUGRC(AUGRC):
    r"""Area Under the SCOD Generalized Risk-Coverage curve.

    Using the same notation as :class:`SCODAURC`, the generalized SCOD risk is:

    .. math::

        \text{SCOD-AUGRC} = \int_0^1 \kappa \cdot r(\kappa)\,\mathrm{d}\kappa
        \approx \frac{1}{N}\sum_{k=1}^{N}\frac{k}{N}\cdot r\!\left(\tfrac{k}{N}\right).
    """

    def update(self, ood_scores: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store SCOD confidence scores and associated detection errors.

        Args:
            ood_scores: OOD scores where higher means more OOD-like.
            targets: Binary labels, with 0 for ID and 1 for OOD.
        """
        self.scores.append(-ood_scores.reshape(-1))
        self.errors.append(targets.reshape(-1).float())


class SCODCovAtxRisk(CovAtxRisk):
    r"""Coverage at x SCOD risk.

    This metric returns the maximum selective coverage for which the SCOD risk
    remains below a target threshold.
    """

    def update(self, ood_scores: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store SCOD confidence scores and associated detection errors.

        Args:
            ood_scores: OOD scores where higher means more OOD-like.
            targets: Binary labels, with 0 for ID and 1 for OOD.
        """
        self.scores.append(-ood_scores.reshape(-1))
        self.errors.append(targets.reshape(-1).float())


class SCODCovAt5Risk(SCODCovAtxRisk):
    r"""Coverage at 5% SCOD risk.

    This is a specific case of :class:`SCODCovAtxRisk` with
    ``risk_threshold = 0.05``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_threshold=0.05, **kwargs)


class SCODRiskAtxCov(RiskAtxCov):
    r"""SCOD risk at x coverage.

    This metric returns the SCOD error rate measured at a fixed selective
    coverage level.
    """

    def update(self, ood_scores: Tensor, targets: Tensor) -> None:  # pyrefly: ignore[bad-override]
        """Store SCOD confidence scores and associated detection errors.

        Args:
            ood_scores: OOD scores where higher means more OOD-like.
            targets: Binary labels, with 0 for ID and 1 for OOD.
        """
        self.scores.append(-ood_scores.reshape(-1))
        self.errors.append(targets.reshape(-1).float())


class SCODRiskAt80Cov(SCODRiskAtxCov):
    r"""SCOD risk at 80% coverage.

    This is a specific case of :class:`SCODRiskAtxCov` with
    ``cov_threshold = 0.8``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(cov_threshold=0.8, **kwargs)
