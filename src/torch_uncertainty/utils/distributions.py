from enum import Enum
from importlib import util
from numbers import Number

import torch
from torch import Tensor
from torch.distributions import (
    Cauchy,
    Distribution,
    Gamma,
    Laplace,
    Normal,
    StudentT,
    constraints,
)
from torch.distributions.utils import broadcast_all

if util.find_spec("scipy"):
    from scipy.special import stdtr, stdtrit

    scipy_installed = True
else:  # coverage: ignore
    scipy_installed = False


class DistEstimate(str, Enum):
    MEAN = "mean"
    MODE = "mode"


def get_dist_class(dist_family: str) -> type[Distribution]:
    """Get the distribution class from a string.

    Args:
        dist_family (str): The distribution family.

    Returns:
        type[Distribution]: The distribution class.
    """
    if dist_family == "normal":
        return Normal
    if dist_family == "laplace":
        return Laplace
    if dist_family == "cauchy":
        return Cauchy
    if dist_family == "gamma":
        return Gamma
    if dist_family == "student":
        return TUStudentT
    if dist_family == "nig":
        return NormalInverseGamma
    raise NotImplementedError(
        f"{dist_family} distribution is currently not supported. Raise an issue if needed."
    )


def get_dist_estimate(dist: Distribution, dist_estimate: DistEstimate) -> Tensor:
    """Get a point-wise prediction from a distribution.

    Args:
        dist (Distribution): The distribution.
        dist_estimate (DistEstimate): The estimate to use.

    Returns:
        Tensor: The estimated value.
    """
    if dist_estimate == DistEstimate.MEAN:
        return dist.mean
    return dist.mode


class TUStudentT(StudentT):
    def cdf(self, value: Tensor) -> Tensor:
        if not scipy_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the distribution option: "
                """pip install -U "torch_uncertainty[distribution]"."""
            )
        if self._validate_args:  # coverage: ignore
            self._validate_sample(value)

        x = ((value - self.loc) / self.scale).detach().cpu().numpy()
        df = self.df.detach().cpu().numpy()

        return torch.tensor(stdtr(df, x), device=self.loc.device)

    def icdf(self, value: Tensor) -> Tensor:
        if not scipy_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the distribution option: "
                """pip install -U "torch_uncertainty[distribution]"."""
            )

        if self._validate_args:  # coverage: ignore
            self._validate_sample(value)

        p = value.detach().cpu().numpy()
        df = self.df.detach().cpu().numpy()

        return torch.tensor(stdtrit(df, p), device=self.loc.device) * self.scale + self.loc


class NormalInverseGamma(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "lmbda": constraints.positive,
        "alpha": constraints.greater_than_eq(1),
        "beta": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        loc: Number | Tensor,
        lmbda: Number | Tensor,
        alpha: Number | Tensor,
        beta: Number | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.lmbda, self.alpha, self.beta = broadcast_all(loc, lmbda, alpha, beta)
        if (
            isinstance(loc, Number)
            and isinstance(lmbda, Number)
            and isinstance(alpha, Number)
            and isinstance(beta, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Impromper mean of the NormalInverseGamma distribution.

        This value is necessary to perform point-wise predictions in the
        regression routine.
        """
        return self.loc

    def mode(self) -> None:
        raise NotImplementedError("NormalInverseGamma distribution has no mode.")

    def stddev(self) -> None:
        raise NotImplementedError("NormalInverseGamma distribution has no stddev.")

    def variance(self) -> None:
        raise NotImplementedError("NormalInverseGamma distribution has no variance.")

    @property
    def mean_loc(self) -> Tensor:
        return self.loc

    @property
    def mean_variance(self) -> Tensor:
        return self.beta / (self.alpha - 1)

    @property
    def variance_loc(self) -> Tensor:
        return self.beta / (self.alpha - 1) / self.lmbda

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:  # coverage: ignore
            self._validate_sample(value)
        gam: Tensor = 2 * self.beta * (1 + self.lmbda)
        return (
            -0.5 * torch.log(torch.pi / self.lmbda)
            + self.alpha * gam.log()
            - (self.alpha + 0.5) * torch.log(gam + self.lmbda * (value - self.loc) ** 2)
            - torch.lgamma(self.alpha)
            + torch.lgamma(self.alpha + 0.5)
        )
