"""
File Name: ITErRoot.py

Authors: Kyle Seidenthal

Date: 30-11-2020

Description: Segmentation network for SPROUT.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
import os
import sys
import collections
import network_modules.evaluation_metrics

from tqdm import tqdm
from network_modules.network_structure import MainNet, SecondaryNet
from torchsummary import summary


class ITErRoot(nn.Module):
    """ Segmentation network for SPROUT """

    def __init__(self, iters):
        """Create the network.

        Args:
            iters (int): The number of iterations (sub-nets) in the network.

        Returns: A fresh network. Only been dropped once.

        """
        super(ITErRoot, self).__init__()

        self.iters = iters

        self.main_net = MainNet(3, 32)

        self.secondary_nets = nn.ModuleList(SecondaryNet(32 * (i+2), out_channels=32)
                                            for i in range(iters))

    def forward(self, x):

        x1, x2, output = self.main_net(x)

        outputs = [output]

        concat_feats = [x1]

        for i in range(self.iters):
            to_concat = concat_feats + [x2]

            x = torch.cat(to_concat, dim=1)

            feats, x2, output = self.secondary_nets[i](x.detach())

            concat_feats.append(feats)

            outputs.append(output)

        return output, outputs
