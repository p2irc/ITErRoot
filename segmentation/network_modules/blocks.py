"""
File Name: blocks.py

Authors: Kyle Seidenthal

Date: 30-11-2020

Description: Definitions of custom building blocks for the network

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    """A basic building block for a Residual CNN."""

    def __init__(self, in_channels, out_channels):
        """Init the Blocks.

        Args:
            in_channels (int): The number of input channels for this block.
            out_channels (int): The number of output channels for this block.

        Returns: A block with the specified layer properties.

        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.relu2 = nn.ReLU()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True)
        )

    def forward(self, x):

        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)

        out = out + identity

        out = self.relu2(out)

        return out


class Down(nn.Module):

    """A single downward block for the network."""

    def __init__(self, in_channels, out_channels):
        """Initialize a downward block.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns: A downward block.

        """
        super(Down, self).__init__()

        self.basic_block_down = nn.Sequential(
                nn.MaxPool2d(2),
                BasicBlock(in_channels, out_channels)
                )

    def forward(self, x):
        return self.basic_block_down(x)


class Up(nn.Module):
    """ A single upward block for the network """

    def __init__(self, in_channels, out_channels):
        """ Initialize an upward block.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns: An upward block.

        """
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)

        self.block = BasicBlock(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x2.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - (diffX // 2),
                        diffY // 2, diffY - (diffY // 2)])

        x = torch.cat([x2, x1], dim=1)

        return self.block(x)


class InputBlock(nn.Module):
    """ An input block."""

    def __init__(self, in_channels, out_channels):
        """Create an input block.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns: An input block.

        """
        super(InputBlock, self).__init__()

        self.block = BasicBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)


class OutputBlock(nn.Module):
    """ An output block """

    def __init__(self, in_channels, out_channels):
        """Create an output block.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        Returns: An output block.

        """
        super(OutputBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        out = self.conv(x)
        out = torch.sigmoid(out)

        return out
