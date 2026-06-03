import torch.nn as nn

from .backbone_utils import forward_with_features, get_classifier, get_fc_numpy


class ReactNet(nn.Module):
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

    def forward_threshold(self, x, threshold):
        _, feature = forward_with_features(self.backbone, x)
        feature = feature.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        return get_classifier(self.backbone)(feature)

    def get_fc(self):
        return get_fc_numpy(self.backbone)
