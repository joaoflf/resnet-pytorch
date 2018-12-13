"""
This module defines the ResNet model (built from scratch)
It contains several classes representing the building blocks of the model,
and a main class ResNet with the ResNet34 model.
The ResNet class is the one to import to get the usable model.
"""

import torch
import torch.nn as nn
DTYPE = torch.cuda.FloatTensor

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

        Returns:
            Tensor: a flattened PyTorch Tensor
        """
        num = x.shape[0]
        return x.view(num, -1)

class ResidualLayer(nn.Module):
    """
    NN module that performs the residual mapping

    This module performs the operation of element-wise sum that takes places
    at the end of each residual block

    Attributes:
        downsampling (boolean): indicates if dimension downsampling occurred
        res_input (Tensor): the residual block input (identity)
    """
    def __init__(self, res_input, downsampling):
        super().__init__()
        self.downsampling = downsampling
        self.res_input = res_input

    def forward(self, x):
        """
        Perform residual mapping

        Does the element-wise sum between the residual block identity and
        the input of this module. If downsampling occurs, perform a 1x1
        convolution to match dimensions and number of channels

        Returns
            Tensor: sum between residual block identity and input
        """
        if self.downsampling:
            shortcut_conv = nn.Sequential(
                nn.Conv2d(self.res_input.shape[1], x.shape[1],
                          kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(x.shape[1])
            ).type(DTYPE)
            output = shortcut_conv(self.res_input) + x
        else:
            output = self.res_input + x
        return output

class ResidualBlock(nn.Module):
    """
    NN module containing a full residual block

    Init the model with the first part of the block, then call the residual
    layer and final relu in the forward method

    Attributes:
        input_channels (int): number of channels in the input
        output_channels (int): number of channels to output
        downsampling (boolean): indicates if dimension downsampling occurs
    """
    def __init__(self, input_channels, output_channels, downsampling=False):
        super().__init__()

        self.downsampling = downsampling
        stride = 2 if downsampling else 1
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        """
        Run the initial part of the residual block, then instantiate and run
        the residual mapping layer and the final relu

        Returns:
            Tensor: final output of the residual block
        """
        output = self.model(x)
        residual_layer = ResidualLayer(x, self.downsampling).type(DTYPE)
        output = residual_layer(output)
        relu = nn.ReLU()
        return relu(output)

class ResNet(nn.Module):
    """
    NN Module containing a full ResNet34 model
    """
    def __init__(self, num_classes):
        super().__init__()
        self.name = 'ResNet34'
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.AvgPool2d(3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, downsampling=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, downsampling=True),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, downsampling=True),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(512, num_classes)
        )
        self.model.apply(self.init_weights)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, layer):
        """
        Initialisation for conv layer weights
        """
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
