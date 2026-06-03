import numpy as np
import torch
import torch.nn as nn

from .backbone_utils import forward_with_features, get_classifier, get_fc_numpy


class ScaleNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, return_feature=False, return_feature_list=False):
        if return_feature_list:
            try:
                return self.backbone(x, return_feature, return_feature_list)
            except TypeError:
                return self.backbone(x, return_feature)
        if return_feature:
            return forward_with_features(self.backbone, x)
        return self.backbone(x)

    def forward_threshold(self, x, percentile):
        _, feature = forward_with_features(self.backbone, x)
        feature = scale(feature.view(feature.size(0), -1, 1, 1), percentile)
        feature = feature.view(feature.size(0), -1)
        return get_classifier(self.backbone)(feature)

    def get_fc(self):
        return get_fc_numpy(self.backbone)


def scale(x, percentile=65):
    x_clone = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2

    return x_clone * torch.exp(scale[:, None, None, None])
