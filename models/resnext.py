import math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    NN module to flatten a tensor from 4D to 2D
    Flattens a 4D tensor outputed from a Convolution (N,C,H,W),
    to a 2D tensor ready to be consumed by a Linear layer (N,C*H*W)
    """

    def forward(self, x):
        """
        Get the dimensions of the input and reshape it to (N,-1)
        Use PyTorch's Tensor view method to reshape the input to (N,C*H*W)
        """
        num = x.shape[0]
        return x.view(num, -1)


CARDINALITY = 32
DEPTH = 4
BASE_WIDTH = 64


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        C = CARDINALITY
        D = int(DEPTH * output_channels / BASE_WIDTH)

        self.model = nn.Sequential(
            nn.Conv2d(
                input_channels, C * D, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(C * D), nn.ReLU(),
            nn.Conv2d(
                C * D,
                C * D,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=C,
                bias=False), nn.BatchNorm2d(C * D), nn.ReLU(),
            nn.Conv2d(C * D, output_channels * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels * 4))

        self.shortcut = nn.Sequential()
        if stride == 2 or input_channels != output_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels * 4,
                    stride=stride,
                    kernel_size=1,
                    bias=False), nn.BatchNorm2d(output_channels * 4))

    def forward(self, x):
        return F.relu(self.model(x) + self.shortcut(x))


class ResNext29(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name = 'ResNext29'
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(64, 64, 1), ResidualBlock(256, 64, 1),
            ResidualBlock(256, 64, 1), ResidualBlock(256, 128, 2),
            ResidualBlock(512, 128, 1), ResidualBlock(512, 128, 1),
            ResidualBlock(512, 256, 2), ResidualBlock(1024, 256, 1),
            ResidualBlock(1024, 256, 1), Flatten(), nn.Linear(
                1024, num_classes))

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
