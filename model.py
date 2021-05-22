import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torchvision import models


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        """
        Initializer of the network class. Define each layer of the network.

        Args:
            A () : ...
        """
        super(Resnet50, self).__init__()
        self.Resnet = models.resnet50(pretrained=True)
        self.Resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        Forward pass of my network.

        Args:
            x () : input of my network

        Returns:
            x () : return of my network 
        """
        return self.Resnet(x)


class EfficientnetV2(nn.Module):
    def __init__(self, num_classes):
        super(EfficientnetV2, self).__init__()
        self.net = timm.create_model('tf_efficientnetv2_s/m/l', pretrained=True)
        self.net.last_linear = nn.Linear(in_features=self.net.last_linear.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)