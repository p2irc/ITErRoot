"""
File Name: compute_data_mean.py

Authors: Kyle Seidenthal

Date: 19-11-2020

Description: A script to compute the mean of the training set, for use in
             normalization for the model.

"""

import argparse
import torch
from network_modules.dataset import SPROUTNetSegDataset
import numpy as np
import os

from torchvision.transforms import Compose
from network_modules.transforms import ToTensor


def main(args):
    dataset = SPROUTNetSegDataset(imgs_dir=os.path.join(args.dataset_path,
                                                        "images"),
                                  masks_dir=os.path.join(args.dataset_path,
                                                         "masks"),
                                  transform=Compose([ToTensor()]))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0,
                                         shuffle=False)

    mean = [0, 0, 0]
    std = [0, 0, 0]

    total_pixels = 0

    for data in loader:
        img = data['image']

        total_pixels += img.shape[2] * img.shape[3]

        mean[0] += img[:, 0, :, :].sum()
        mean[1] += img[:, 1, :, :].sum()
        mean[2] += img[:, 2, :, :].sum()

    mean[0] = torch.true_divide(mean[0], total_pixels)
    mean[1] = torch.true_divide(mean[1], total_pixels)
    mean[2] = torch.true_divide(mean[2], total_pixels)

    for data in loader:
        img = data['image']

        std[0] += ((img[:, 0, :, :] - mean[0]) ** 2).sum()
        std[1] += ((img[:, 1, :, :] - mean[1]) ** 2).sum()
        std[2] += ((img[:, 2, :, :] - mean[2]) ** 2).sum()

    std[0] = np.sqrt(torch.true_divide(std[0], total_pixels))
    std[1] = np.sqrt(torch.true_divide(std[1], total_pixels))
    std[2] = np.sqrt(torch.true_divide(std[2], total_pixels))

    print("Mean: {}".format(mean))
    print("STD: {}".format(std))


def get_args():

    parser = argparse.ArgumentParser(description="Find the mean and std of"
                                                 " the dataset.")

    parser.add_argument("dataset_path",
                        type=str,
                        help="Path to the directory containing the images/ "
                             "and masks/ directories.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
