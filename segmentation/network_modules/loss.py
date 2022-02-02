"""
File Name: loss.py

Authors: Kyle Seidenthal

Date: 16-10-2020

Description: Custom loss functions for SPROUTNet Seg
             |  |!
             || |-
"""
import torch
import network_modules.evaluation_metrics as em
import torch.nn.functional as F

from torch.autograd import Variable
from torch import nn


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class CombinedDLCrossent(nn.Module):
    """
    Dice Loss + BinaryCrossentropy Loss
    """

    def __init__(self):

        super(CombinedDLCrossent, self).__init__()

        self.crossent_loss = nn.BCELoss()
        self.gdl_loss = DiceLoss()

    def forward(self, predicted, target):

        return (self.crossent_loss(predicted.float(),
                target.unsqueeze(1).float()) +
                self.gdl_loss(predicted, target))


class CombinedWDLCrossent(nn.Module):
    """
    (dice_weight * Dice Loss) + (crossent_weight + BinaryCrossentropy)

    Attributes:
        {% An Attribute %}: {% Description %}
    """

    def __init__(self, dice_weight=1.0, crossent_weight=1.0):

        super(CombinedWDLCrossent, self).__init__()

        self.crossent_loss = nn.BCELoss()
        self.gdl_loss = DiceLoss()

        self.dice_weight = dice_weight
        self.crossent_weight = crossent_weight

    def forward(self, predicted, target):

        return ((self.crossent_weight *
                self.crossent_loss(predicted.float(),
                                   target.unsqueeze(1).float())) +
                (self.dice_weight * self.gdl_loss(predicted, target)))


class CombinedIoUCrossent(nn.Module):
    def __init__(self):
        super(CombinedIoUCrossent, self).__init__()

        self.crossent_loss = nn.BCELoss()
        self.iou_loss = IoULoss()

    def forward(self, predicted, target):

        return (self.crossent_loss(predicted.float(),
                                   target.unsqueeze(1).float()) +
                self.iou_loss(predicted, target))


class DiceLoss(nn.Module):
    """
    Dice Loss
    """
    def __init__(self, smooth=1e-5):
        """

        Args:
            smooth (float): Amount of smoothing to use to avoid division by 0

        """
        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, predicted, target):

        predicted = predicted.view(-1)
        target = target.view(-1)

        intersect = (predicted * target).sum()

        union = (predicted.sum() + target.sum() + self.smooth)

        dice = (2. * intersect + self.smooth) / union

        return 1 - dice


class IoULoss(nn.Module):
    """
    IoU Loss
    """

    def __init__(self, smooth=1e-5):
        """Loss

        Kwargs:
            smooth (float): Amount of smoothing to use to avoid division by 0.

        """
        super(IoULoss, self).__init__()

        self.smooth = smooth

    def forward(self, predicted, target):
        predicted = predicted.view(-1)
        target = target.view(-1)

        intersect = (predicted * target).sum()

        union = predicted.sum() + target.sum() - intersect + self.smooth

        iou = intersect / union

        return 1 - iou
