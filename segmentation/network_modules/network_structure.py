"""
File Name: network_structure.py

Authors: Kyle Seidenthal

Date: 30-11-2020

Description: Definition of basic network pieces.
"""

import network_modules.blocks as blocks
import torch
import torch.nn as nn


class MainNet(nn.Module):
    """ The Main larger Unet Structure"""

    def __init__(self, n_channels, out_channels):
        """Init a main network.

        Args:
            n_channels (int): The number of input channels.
            out_channels (int): The number of channels to output.

        Returns: The network.

        """

        super(MainNet, self).__init__()

        self.n_channels = n_channels
        self.out_channels = out_channels

        # Layers
        self.input = blocks.InputBlock(n_channels, out_channels)

        self.down1 = blocks.Down(out_channels, out_channels * 2)
        self.down2 = blocks.Down(out_channels * 2, out_channels * 4)
        self.down3 = blocks.Down(out_channels * 4, out_channels * 8)
        self.down4 = blocks.Down(out_channels * 8, out_channels * 16)

        self.up1 = blocks.Up(out_channels * 16, out_channels * 8)
        self.up2 = blocks.Up(out_channels * 8, out_channels * 4)
        self.up3 = blocks.Up(out_channels * 4, out_channels * 2)
        self.up4 = blocks.Up(out_channels * 2, out_channels)

        self.output = blocks.OutputBlock(out_channels, 1)

    def forward(self, x):

        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.output(x)

        return x1, x, out


class SecondaryNet(nn.Module):
    """ The secondary smaller Unet Structure"""

    def __init__(self, n_channels, out_channels):
        """Init a secondary network.

        Args:
            n_channels (int): The number of input channels.
            out_channels (int): The number of channels to output.

        Returns: The network.

        """

        super(SecondaryNet, self).__init__()

        self.n_channels = n_channels
        self.out_channels = out_channels

        # Layers
        self.input = blocks.InputBlock(n_channels, out_channels)

        self.down1 = blocks.Down(out_channels, out_channels * 2)
        self.down2 = blocks.Down(out_channels * 2, out_channels * 4)
        self.down3 = blocks.Down(out_channels * 4, out_channels * 8)

        self.up1 = blocks.Up(out_channels * 8, out_channels * 4)
        self.up2 = blocks.Up(out_channels * 4, out_channels * 2)
        self.up3 = blocks.Up(out_channels * 2, out_channels)

        self.output = blocks.OutputBlock(out_channels, 1)

    def forward(self, x):

        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        out = self.output(x)

        return x1, x, out
