from collections.abc import Callable

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class SGLD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        noise_factor: float = 1e-2,
        weight_decay: float = 0,
    ) -> None:
        r"""Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

        Note:
            Use torch_uncertainty.methods.CheckpointCollector to collect the samples.

        Args:
            params (ParamsT): Iterable of parameters or named_parameters to optimize or iterable of
                dicts defining parameter groups. When using named_parameters, all parameters in all
                groups should be named.
            lr (float, optional): Learning rate for the optimization. Defaults to ``1e-3``.
            noise_factor (float, optional): A factor to reduce the amount of noise and stabilize the training.
                This parameter was not proposed in the original paper. Defaults to ``1e-2``.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to ``0``.

        Reference:
            - `Bayesian Learning via Stochastic Gradient Langevin Dynamics <https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf>`_.
        """
        defaults = {"lr": lr, "noise_factor": noise_factor, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            noise_factor = group["noise_factor"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # original gradient
                d_p = p.grad.data

                # Centered Gaussian assumption
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # Noise of variance 2*lr
                noise = torch.randn_like(p) * ((2 * lr) ** 0.5) * noise_factor

                # Weight update
                p.data.add_(-lr * d_p + noise)

        return loss
