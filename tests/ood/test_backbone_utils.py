import torch
from torch import nn

from torch_uncertainty.models.classification.resnet.std import _BasicBlock, _ResNet
from torch_uncertainty.ood.nets import ASHNet
from torch_uncertainty.ood.nets.backbone_utils import (
    forward_with_features,
    get_fc_numpy,
    model_supports_ood_features,
    supports_return_feature,
)


def _tiny_resnet(num_classes: int = 10) -> _ResNet:
    return _ResNet(
        block=_BasicBlock,
        num_blocks=[1, 1, 1],
        in_channels=3,
        num_classes=num_classes,
        conv_bias=False,
        dropout_rate=0.0,
        groups=1,
        style="cifar",
    )


def test_resnet_supports_ood_features_and_return_feature():
    model = _tiny_resnet()
    assert model_supports_ood_features(model)
    assert supports_return_feature(model)
    x = torch.randn(2, 3, 32, 32)
    logits, feat = forward_with_features(model, x)
    assert logits.shape == (2, 10)
    assert feat.shape == (2, model.feature_size)


def test_resnet_ash_net_forward_threshold():
    model = _tiny_resnet()
    x = torch.randn(2, 3, 32, 32)
    ash = ASHNet(model)
    out = ash.forward_threshold(x, percentile=65)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_feats_forward_fallback_without_return_feature_api():
    class _BareModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear = nn.Linear(4, 3)

        def feats_forward(self, x):
            return self.pool(self.conv(x)).flatten(1)

        def forward(self, x):
            return self.linear(self.feats_forward(x))

    model = _BareModel()
    assert model_supports_ood_features(model)
    assert not supports_return_feature(model)
    logits, feat = forward_with_features(model, torch.randn(1, 3, 8, 8))
    assert logits.shape == (1, 3)
    assert feat.shape == (1, 4)
    w, b = get_fc_numpy(model)
    assert w.shape == (3, 4)
    assert b.shape == (3,)
