from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.compute import _auc_compute
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE


class AUSE(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 100.0
    plot_legend_name = "Sparsification Curves"

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, **kwargs) -> None:
        r"""The Area Under the Sparsification Error curve (AUSE) metric to evaluate
        the quality of the uncertainty estimates, i.e., how much they coincide with
        the true errors.

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`scores`: Uncertainty scores of shape :math:`(B,)`. A higher
              score means a higher uncertainty.
            - :attr:`errors`: Errors of shape :math:`(B,)`,

            where :math:`B` is the batch size.

        References:
            [1] `Uncertainty estimates and multi-hypotheses for optical flow. In ECCV, 2018
            <https://arxiv.org/abs/1802.07095>`_.

        Note:
            A higher AUSE means a lower quality of the uncertainty estimates.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, scores: Tensor, errors: Tensor) -> None:
        """Store the scores and their associated errors for later computation.

        Args:
            scores (Tensor): uncertainty scores of shape :math:`(B,)`
            errors (Tensor): errors of shape :math:`(B,)`
        """
        self.scores.append(scores)
        self.errors.append(errors)

    def partial_compute(self) -> tuple[Tensor, Tensor]:
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        if scores.shape[0] < 2:
            nan = torch.tensor([float("nan")], device=self.device)
            return nan, nan
        error_rates = _ause_rejection_rate_compute(scores, errors)
        optimal_error_rates = _ause_rejection_rate_compute(errors, errors)
        return error_rates.cpu(), optimal_error_rates.cpu()

    def compute(self) -> Tensor:
        """Compute the Area Under the Sparsification Error curve (AUSE) based
        on inputs passed to ``update``.

        Returns:
            Tensor: The AUSE.
        """
        error_rates, optimal_error_rates = self.partial_compute()
        if torch.isnan(error_rates[0]).item():
            return torch.tensor([float("nan")], device=self.device)
        num_samples = error_rates.size(0)
        x = torch.arange(0, num_samples, device=self.device) / num_samples
        y = error_rates - optimal_error_rates
        return torch.tensor([_auc_compute(x, y)])

    def plot(
        self,
        ax: _AX_TYPE | None = None,
        plot_oracle: bool = True,
        plot_value: bool = True,
    ) -> tuple[Figure | None, Axes | object]:
        """Plot the sparsification curve corresponding to the inputs passed to
        ``update``, and the oracle sparsification curve.

        Args:
            ax (Axes | None, optional): An matplotlib axis object. If provided
                will add plot to this axis. Defaults to None.
            plot_oracle (bool, optional): Whether to plot the oracle
                sparsification curve. Defaults to True.
            plot_value (bool, optional): Whether to plot the AUSE value.
                Defaults to True.

        Returns:
            tuple[[Figure | None], Axes]: Figure object and Axes object
        """
        fig, ax = plt.subplots() if ax is None else (None, ax)

        # Computation of AUSEC
        error_rates, optimal_error_rates = self.partial_compute()
        num_samples = error_rates.size(0)
        x = torch.arange(num_samples) / num_samples
        y = error_rates - optimal_error_rates

        ausec = _auc_compute(x, y).cpu().item()

        rejection_rates = torch.arange(num_samples) / num_samples * 100

        ax.plot(
            rejection_rates,
            error_rates * 100,
            label="Model",
        )
        if plot_oracle:
            ax.plot(
                rejection_rates,
                optimal_error_rates * 100,
                label="Oracle",
            )

        ax.set_xlabel("Rejection Rate (%)")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xlim(self.plot_lower_bound, self.plot_upper_bound)
        ax.set_ylim(self.plot_lower_bound, self.plot_upper_bound)
        ax.legend(loc="upper right")

        if plot_value:
            ax.text(
                0.02,
                0.02,
                f"AUSEC={ausec:.03}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )

        return fig, ax


def _ause_rejection_rate_compute(
    scores: Tensor,
    errors: Tensor,
) -> Tensor:
    """Compute the cumulative error rates for a given set of scores and errors.

    Args:
        scores (Tensor): uncertainty scores of shape :math:`(B,)`
        errors (Tensor): errors of shape :math:`(B,)`
    """
    num_samples = errors.size(0)

    order = scores.argsort()
    errors = errors[order]

    error_rates = torch.zeros(num_samples + 1)
    error_rates[0] = errors.sum()
    error_rates[1:] = errors.cumsum(dim=-1).flip(0)
    return error_rates / error_rates[0]
