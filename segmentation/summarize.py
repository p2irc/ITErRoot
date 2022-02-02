"""
File Name: summarize.py

Authors: {% <AUTHOR> %}

Date: 13-03-2021

Description: {% <DESCRIPTION> %}

"""
import torch
import json
import csv
import argparse
import hypertune

import os
import numpy as np
import tarfile
import network_modules.dataset as data

from network_modules.loss import (CombinedDLCrossent, DiceLoss, IoULoss,
                                  CombinedIoUCrossent, CombinedWDLCrossent)
from network_modules.sprout_net_seg import SproutNetSeg
from network_modules.dataset import SPROUTNetSegDataset
from network_modules.transforms import (RandomFlip, RandomRotation, ToTensor,
                                        Normalize)
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from network_modules.early_stopping import EarlyStopperBase, EarlyStopperSingle
from torchsummary import summary
from tqdm import tqdm
from network_modules import evaluation_metrics as em
from google.cloud import storage

model = SproutNetSeg(3)

summary(model, (3, 256, 256))
