from collections import OrderedDict

import torch
from torch import nn


class Resnet(nn.Module):

    def __init__(self, num_channels: int):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([('relu1', nn.ReLU()),
                                                ('conv1', nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)),
                                                ('relu2', nn.ReLU()),
                                                ('conv2', nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1))]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class ConvSequence(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([('conv',
                                                 nn.Conv2d(in_channels=in_num_channels, out_channels=out_num_channels, kernel_size=3, padding=1)),
                                                ('maxpool', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                                                ('resnet', Resnet(out_num_channels)),
                                                ('resnet', Resnet(out_num_channels))]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ImpalaCNN(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([('ImpalaConvSeq1', ConvSequence(3, depths[0])),
                                                ('ImpalaConvSeq2', ConvSequence(depths[0], depths[1])),
                                                ('ImpalaConvSeq3', ConvSequence(depths[1], depths[2])),
                                                ('ImpalaFlatten', nn.Flatten(1, -1)),
                                                ('ImpalaRelu1', nn.ReLU()),
                                                ('ImpalaFC', nn.Linear(2048, 256)),
                                                ('ImpalaRelu2', nn.ReLU())]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
