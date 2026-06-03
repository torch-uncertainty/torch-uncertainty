import torch
import torch.nn as nn

from .backbone_utils import forward_with_features, get_classifier


class AdaScaleANet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.logit_scaling = False

    def forward(self, x, return_feature=False, return_feature_list=False):
        if return_feature_list:
            try:
                return self.backbone(x, return_feature, return_feature_list)
            except TypeError:
                return self.backbone(x, return_feature)
        if return_feature:
            return forward_with_features(self.backbone, x)
        return self.backbone(x)

    def forward_threshold(self, feature, percentiles):
        scale = ada_scale(torch.relu(feature), percentiles)
        classifier = get_classifier(self.backbone)
        if self.logit_scaling:
            logits_cls = classifier(feature)
            logits_cls *= scale**2.0
        else:
            feature *= torch.exp(scale)
            logits_cls = classifier(feature)
        return logits_cls


class AdaScaleLNet(AdaScaleANet):
    def __init__(self, backbone):
        super().__init__()
        self.logit_scaling = True


def ada_scale(x, percentiles):
    assert x.dim() == 2
    b, c = x.shape
    assert percentiles.shape == (b,)
    assert x.dim() == 2, "input tensor must be 2D"
    assert torch.all(percentiles > 0), "percentiles must be > 0"
    assert torch.all(percentiles < 100), "percentiles must be < 100"
    n = x.shape[1:].numel()
    ks = n - torch.round(n * percentiles.cuda() / 100.0).to(torch.int)
    max_k = ks.max()
    values, _ = torch.topk(x, max_k, dim=1)
    mask = torch.arange(max_k, device=x.device)[None, :] < ks[:, None]
    batch_sums = x.sum(dim=1, keepdim=True)
    masked_values = values * mask
    topk_sums = masked_values.sum(dim=1, keepdim=True)
    return batch_sums / topk_sums
