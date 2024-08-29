import torch.nn as nn
import torchvision
from mmdet3d.models.builder import BACKBONES


@BACKBONES.register_module()
class BackboneNet(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet101()
        self.net = nn.Sequential(
            *list(resnet.children())[:-6], resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        ).cuda()

    def forward(self, x):
        return [self.net(x)]
