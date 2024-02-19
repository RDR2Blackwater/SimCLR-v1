import torch.nn as nn
from torchvision import models

from backbones.resnet_series import SimCLR_ResNet50


class SimCLR_v1(nn.Module):
    def __init__(self, backbone, width: int, features: int = 2048):
        """
        :param backbone: Backbone of SimCLR
        :param features: Representation dimension of non-linear projection head
        :param width: Dimension multiplier, expand the hidden dimension of ResNet
        """
        super(SimCLR_v1, self).__init__()
        # define backbone of SimCLR
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=features),
                            "resnet50": models.resnet50(weights=None, num_classes=features),
                            "simclr_resnet50": SimCLR_ResNet50(width_multiplier=width, num_classes=features)}

        self.backbone = self._get_basemodel(backbone)
        dim_mlp = self.backbone.fc.in_features

        # add 2-layer mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                         nn.ReLU(),
                                         self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise KeyError("Invalid backbone architecture. Check the config and pass one of: resnet18/resnet50/simclr_resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
