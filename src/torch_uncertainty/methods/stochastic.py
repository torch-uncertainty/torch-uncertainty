import torch
from torch import Tensor, nn

from torch_uncertainty.layers.bayesian import bayesian_modules


class StochasticModel(nn.Module):
    def __init__(
        self,
        core_model: nn.Module,
        num_samples: int,
        probabilistic: bool = False,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.num_samples = num_samples
        self.probabilistic = probabilistic

    def eval_forward(self, x: Tensor) -> Tensor:
        out = [self.core_model(x) for _ in range(self.num_samples)]
        if self.probabilistic:
            key_set = {tuple(o.keys()) for o in out}
            return {k: torch.cat([o[k] for o in out], dim=0) for k in key_set.pop()}
        return torch.cat(out, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)

    def sample(self, num_samples: int = 1) -> list[dict]:
        sampled_models = [{}] * num_samples
        for module_name in self.core_model._modules:
            module = self.core_model._modules[module_name]
            if isinstance(module, bayesian_modules):
                for model in sampled_models:
                    weight, bias = module.sample()
                    model[module_name + ".weight"] = weight
                    if bias is not None:
                        model[module_name + ".bias"] = bias
            else:
                for model in sampled_models:
                    state = module.state_dict()
                    if not len(state):  # no parameter
                        break
                    # TODO: fix this
                    model |= {
                        module_name + "." + key: val for key, val in module.state_dict().items()
                    }
        return sampled_models

    def freeze(self) -> None:
        for module in self.core_model.modules():
            if isinstance(module, bayesian_modules):
                module.freeze()

    def unfreeze(self) -> None:
        for module in self.core_model.modules():
            if isinstance(module, bayesian_modules):
                module.unfreeze()
