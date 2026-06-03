from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

from torch_uncertainty.ood.ood_criteria import NECOCriterion, get_ood_criterion
from torch_uncertainty.ood.utils import ConfigNamespace, load_config

_CONFIG = Path(__file__).resolve().parents[2] / "torch_uncertainty/ood/configs/neco.yml"


class _FeatureModel(nn.Module):
    def __init__(self, feat_dim: int = 32, num_classes: int = 5):
        super().__init__()
        self.feature_size = feat_dim
        self.linear = nn.Linear(feat_dim, num_classes)

    def feats_forward(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feat = self.feats_forward(x)
        logits = self.linear(feat)
        if return_feature:
            return logits, feat
        return logits


def _fit_dummy_setup(crit: NECOCriterion, train_feats: np.ndarray) -> None:
    crit.scaler = StandardScaler()
    train_scaled = crit.scaler.fit_transform(train_feats)
    crit.pca = PCA(n_components=train_scaled.shape[1])
    crit.pca.fit(train_scaled)
    crit.setup_flag = True


def test_get_ood_criterion_loads_neco():
    crit = get_ood_criterion("neco")
    assert isinstance(crit, NECOCriterion)
    assert crit.neco_dim == 100


def test_neco_ratio_score_separates_id_from_ood():
    config = load_config(str(_CONFIG))
    config.postprocessor.postprocessor_args.neco_dim = 4
    config.postprocessor.postprocessor_args.scale_by_maxlogit = False
    crit = NECOCriterion(config)

    rng = np.random.default_rng(0)
    train_id = rng.normal(size=(200, 16)).astype(np.float32)
    train_id[:, :4] += 2.0
    _fit_dummy_setup(crit, train_id)

    id_sample = train_id[:8]
    ood_sample = rng.normal(size=(8, 16)).astype(np.float32) * 0.1
    zeros = np.zeros((8, 5), dtype=np.float32)

    assert crit._ratio_score(id_sample, zeros).mean() > crit._ratio_score(ood_sample, zeros).mean()


def test_neco_forward_integration():
    config = ConfigNamespace(
        {
            "postprocessor": {
                "postprocessor_args": ConfigNamespace(
                    {
                        "neco_dim": 4,
                        "use_scaler": True,
                        "scale_by_maxlogit": False,
                    }
                ),
                "postprocessor_sweep": ConfigNamespace({"neco_dim": [4]}),
            }
        }
    )
    crit = NECOCriterion(config)
    model = _FeatureModel(feat_dim=16, num_classes=5)
    _fit_dummy_setup(crit, np.random.randn(32, 16).astype(np.float32))

    out = crit(model, torch.randn(4, 16))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
