from collections.abc import Callable

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class SGHMC(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        friction: float = 0.05,
        burn_in_steps: int = 200,
        noise_factor: float = 1e-2,
        weight_decay: float = 0,
    ) -> None:
        r"""Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) optimizer.

        Note:
            Use torch_uncertainty.methods.CheckpointCollector to collect the samples.

        Args:
            params (ParamsT): Iterable of parameters or named_parameters to optimize or iterable of
                dicts defining parameter groups. When using named_parameters, all parameters in all
                groups should be named.
            lr (float): Learning rate :math:`\epsilon`. Defaults to ``1e-2``.
            burn_in_steps (int): Number of discarded steps used to update the state. Defaults to ``200``.
            friction (float, optional): The friction term :math:`C`. Defaults to ``0.05``.
            noise_factor (float): A factor to reduce the amount of noise and stabilize the training.
                This parameter was not proposed in the original paper. Defaults to ``1e-2``.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to ``0``.

        Reference:
            - [1] `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102>`_.
            - [2] `Bayesian Optimization with Robust Bayesian Neural Networks <https://papers.nips.cc/paper/6117-bayesian-optimization-with-robust-bayesian-neural-networks.pdf>`_.
            - Code inspired from this `repository <https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_HMC_SA/optimizers.py>`_.
        """
        self.eps = 1e-6
        defaults = {
            "lr": lr,
            "friction": friction,
            "noise_factor": noise_factor,
            "weight_decay": weight_decay,
        }
        self.burn_in_steps = burn_in_steps
        self.burnt_in = torch.tensor(0, dtype=torch.long)
        super().__init__(params, defaults)

    def step(
        self, closure: Callable[[], float] | None = None, burn_in: bool = False
    ) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            friction = group["friction"]
            weight_decay = group["weight_decay"]
            noise_factor = group["noise_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # define dict for each individual parameter
                if len(state) == 0:
                    # estimate of the uncentered variance of the gradient, Eq. 8 in [2]
                    state["v_hat"] = torch.ones_like(p)
                    # free parameter vector specifying the exponential average window, Eq. 8 in [2]
                    state["tau"] = torch.ones_like(p)
                    # smoothed gradient estimate, Eq. 9 in [2]
                    state["g"] = torch.ones_like(p)
                    state["momentum"] = torch.zeros_like(p)

                tau = state["tau"]
                v_hat = state["v_hat"]
                g = state["g"]
                momentum = state["momentum"]

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if self.burnt_in < self.burn_in_steps:
                    # update the moving average window according to Eq. 9 in [2] left equation
                    tau.add_(-(tau * g**2) / (v_hat + self.eps) + 1)
                    # compute tau inverse
                    tau_inv = 1 / (tau + self.eps)
                    # update the smoothed gradient estimate, Eq. 9 in [2] right equation
                    g.add_(-tau_inv * g + tau_inv * d_p)
                    # update the uncentered variance of the gradient, Eq. 8 in [2]
                    v_hat.add_(-tau_inv * v_hat + tau_inv * d_p**2)
                    self.burnt_in += 1

                v_inv_sqrt = 1 / (v_hat.sqrt() + self.eps)

                noise_var = 2 * lr**2 * v_inv_sqrt * friction - lr**4
                noise_std = torch.clamp(noise_var, min=1e-8 / noise_factor**2).sqrt() * noise_factor

                # sample random noise
                noise_sample = torch.normal(
                    mean=torch.zeros_like(d_p), std=torch.ones_like(d_p) * noise_std
                )

                # update momentum, Eq. 10 in [2] right equation
                momentum.add_(-(lr**2) * v_inv_sqrt * d_p - friction * momentum + noise_sample)

                # update parameter, Eq. 10 in [2] left equation
                p.data.add_(momentum)

        return loss
