"""
File Name: dataset.py

Authors: Kyle Seidenthal

Date: 04-07-2021

Description: Dataseat dataloader for ITErRoot.

"""

import os
import torch
import tarfile
import sys

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from torch.utils.data import Dataset
from google.cloud import storage
from network_modules.transforms import (RandomFlip, RandomRotation, ToTensor,
                                        Normalize)
from network_modules import patch_operations

from torchvision.transforms import Compose
from torch.utils.data import DataLoader


NORM_MEAN = [0.2366, 0.2366, 0.2366]
NORM_STD = [0.1814, 0.1669, 0.1498]

class ITErRootTrainingDataset(Dataset):

    """2D Root image dataset for ITErRoot"""

    def __init__(self, imgs_dir, masks_dir, transform=None):
        """

        Args:
            imgs_dir (string): The path to the directory containing the images.
            masks_dir (string): The path to the directory containing the
                                masks.

        Kwargs:
            transform: Optional transform to be applied to a sample.

        """
        Dataset.__init__(self)

        self._imgs_dir = imgs_dir
        self._masks_dir = masks_dir
        self._transform = transform

        self._imgs = sorted(os.listdir(self._imgs_dir))
        self._masks = sorted(os.listdir(self._masks_dir))

    def __len__(self):
        """Get the length of the dataset.

        Returns: The length of the dataset.

        """
        return len(self._imgs)

    def __getitem__(self, idx):
        """Return the item at the given index.

        Args:
            idx (integer): The index in the dataset to get data from.

        Returns: The item at the given index.

        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self._imgs_dir, self._imgs[idx])
        mask_name = os.path.splitext(self._imgs[idx])[0] + "_mask.png"
        mask_name = os.path.join(self._masks_dir, mask_name)

        assert os.path.exists(mask_name)
        assert os.path.exists(img_name)
        assert os.path.splitext(self._imgs[idx])[0] in mask_name

        image = io.imread(img_name)
        mask = io.imread(mask_name).astype(np.bool)

        if image.shape != (256, 256, 3):
            print(img_name, image.shape)

        sample = {'image': image, 'mask': mask}

        if self._transform:
            sample = self._transform(sample)

        sample["img_name"] = img_name

        return sample

class ITErRootPredictDataset(Dataset):

    """2D Root image dataset for ITErRoot"""

    def __init__(self, imgs_dir, transform=None):
        """

        Args:
            imgs_dir (string): The path to the directory containing the images.

        Kwargs:
            transform: Optional transform to be applied to a sample.

        """
        Dataset.__init__(self)

        self._imgs_dir = imgs_dir
        self._transform = transform

        self._imgs = sorted(os.listdir(self._imgs_dir))

    def __len__(self):
        """Get the length of the dataset.

        Returns: The length of the dataset.

        """
        return len(self._imgs)

    def __getitem__(self, idx):
        """Return the item at the given index.

        Args:
            idx (integer): The index in the dataset to get data from.

        Returns: The item at the given index.

        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self._imgs_dir, self._imgs[idx])

        assert os.path.exists(img_name)

        image = io.imread(img_name)

        if image.shape != (256, 256, 3):
            print(img_name, image.shape)

        sample = {'image': image}

        if self._transform:
            sample = self._transform(sample)

        sample["img_name"] = img_name

        return sample

def download_from_gloud(dataset_name, storage_bucket_id):
    """ Download dataset tar from google cloud storage.

    Args:
        dataset_name (string): The name of the dataset file to download.
        storage_bucket_id (string): Google cloud storage bucket ID.

    Returns: The path to the downloaded dataset.

    """
    print("Downloading {}...".formag(dataset_name))

    client = storage.Client()

    bucket = client.get_bucket(storage_bucket_id)

    out_path = os.path.join(".", dataset_name)

    data_blob = bucket.blob(dataset_name)
    data_blob.download_to_filename(out_path)

    tar = tarfile.open(out_path)
    tar.extraxtall()
    tar.close()

    print("Done!")

    extracted_path = os.path.join(".", dataset_path.split(".tar.gz")[0])

    return extracted_path


def load_patch_set(path, seed, batch_size, training=False):
    """Load a patch dataset.

    Args:
        path (string): The path to the dataset.
        seed (int): Random seed to use.
        batch_size (int): Batch size for loading images.

    Kwargs:
        training (bool): Whether to use a training dataloader.  If true, the
        dataset path must contain an images/ and masks/ directory where masks/
        contains the ground truth.

    Returns: A dataloader

    """
    norm = Normalize(NORM_MEAN, NORM_STD)

    if training:
        img_path = os.path.join(path, "images")
        mask_path = os.path.join(path, "masks")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print("Dataset path must include images/ and masks/ directories "
                  "for training=True")

            sys.exit(1)

        transforms = Compose([RandomFlip(), RandomRotation(),
                              ToTensor(), norm])
        data_set = ITErRootTrainingDataset(imgs_dir=img_path,
                                           masks_dir=mask_path,
                                           transform=transforms)
    else:
        img_path = path

        transforms = Compose([ToTensor(), norm])

        data_set = ITErRootPredictDataset(imgs_dir=img_path,
                                          transform=transforms)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    return data_loader


def load_full_set(path, seed, batch_size):
    """ Load a full image set for prediction.

    Args:
        path (string): The path to the images.
        seed (int): Random seed.
        batch_size (int): The batch size for prediction.

    Returns: A data loader.

    """

    # Create patches
    patch_path = os.path.join(path, "patches")

    os.makedirs(patch_path)

    patch_operations.create_patches(path, patch_path)

    norm = Normalize(NORM_MEAN, NORM_STD)

    transforms = Compose([ToTensor(), norm])

    data_set = ITErRootPredictDataset(imgs_dir=patch_path,
                                      transform=transforms)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)


    return data_loader
