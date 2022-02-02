"""
File Name: predict.py

Authors: Kyle Seidenthal

Date: 04-07-2021

Description: A script which will generate predicted segmentations for all
             images in a given dataset.

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
from network_modules.ITErRoot import ITErRoot
from network_modules.dataset import ITErRootPredictDataset
from network_modules.transforms import (RandomFlip, RandomRotation, ToTensor,
                                        Normalize)
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from network_modules.early_stopping import EarlyStopperBase, EarlyStopperSingle
from torchsummary import summary
from tqdm import tqdm
from network_modules import evaluation_metrics as em
from google.cloud import storage
from shutil import copyfile, copytree
from torchvision.utils import save_image

from skimage.util import img_as_float, img_as_bool
from skimage.color import rgb2gray
from network_modules import patch_operations
from sklearn.metrics import confusion_matrix

STORAGE_BUCKET_ID = "your-storage-bucket-id"

def _set_devices(model):
    """Set the device for the model.

    Args:
        model (model): The model to set up.

    Returns: The model on the correct device.

    """

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        print("Using CPU")
        model = model.cpu()

    return model


def _load_model(checkpoint_path):
    """
    Load the model from a checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint.

    Returns:
        The loaded model, the epoch the model was saved at.
    """

    checkpoint = torch.load(checkpoint_path)

    model = ITErRoot(checkpoint['iterations'])

    model = _set_devices(model)

    model.load_state_dict(checkpoint["state"])

    return model, checkpoint['epoch']


def get_model_from_gcloud(gcloud_checkpoint_path, out_path):

    print("Loading Model From GCloud...")
    client = storage.Client()

    bucket = client.get_bucket(STORAGE_BUCKET_ID)

    data_blob = bucket.blob(gcloud_checkpoint_path)

    data_blob.download_to_filename(out_path)
    print("Done")


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a model '
                                                 'checkpoint.'
                                                 'This can be done with Google'
                                                 'Cloud Services using the '
                                                 'corresponding flags.  If '
                                                 'the --gcloud flag is not '
                                                 'set, you must specify the '
                                                 '--dataset-path flag.')

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="The random pytorch seed.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="The batch size to use.")

    parser.add_argument("--checkpoint-path",
                        type=str,
                        help="The path to the model checkpoint.")

    parser.add_argument('--gcloud-checkpoint-path',
                        type=str,
                        help="The path to the checkpoint in GCloud.")

    parser.add_argument('eval_out',
                        type=str,
                        help="Where to put output images")

    parser.add_argument('--dataset_path',
                        type=str,
                        help="The path to the dataset to use if not using"
                             "Gcloud.")

    parser.add_argument('--gcloud',
                        action="store_true",
                        help="Add this flag to use gcloud.")

    parser.add_argument("--patch_set",
                        action="store_true",
                        help="Add this flag to predict for a set of patches"
                             " rather than full images.")

    parser.add_argument("--gcloud_set_name",
                        type=str,
                        help="The name of the dataset file to download from"
                        " gcloud.")

    parser.add_argument("--storage_bucket_id",
                        type=str,
                        help="The storage bucket id for the dataset to "
                        " download from gcloud.")

    args = parser.parse_args()

    return args

def predict_patches(dataloader, model, eval_out):
    """Predict the segmentations for patch images.

    Args:
        dataloader (DataLoader): The dataloader of patch images.
        model (model): The trained model to use.
        eval_out (string): Path to a place to store predictions.

    Returns: None

    """
    if not os.path.exists(eval_out):
        os.makedirs(eval_out)

    if isinstance(model, torch.nn.DataParallel):
        iters = model.module.iters

    else:
        iters = model.iters

    model.eval()

    with tqdm(total=len(dataloader), desc=f'Predicting...',
              unit='batch') as pbar:

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                if torch.cuda.is_available():
                    imgs = data['image'].cuda()

                else:
                    imgs = data['image']

                output, outputs = model(imgs)

                pred_bool = (output > 0.5).float()

                for j in range(imgs.shape[0]):
                    pred_name = os.path.split(data['img_name'][j])[-1].replace(".png",
                                                                   "_mask.png")
                    pred_name = os.path.join(eval_out, pred_name)
                    output_bool = (output[j] > 0.5).float()

                    save_image(output_bool, pred_name)
                pbar.update(1)

def predict_full(dataloader, model, eval_out):
    """Predict the segmentations for full size images.

    Args:
        dataloader (DataLoader): The dataloader for the images.
        model (model): The trained model to use.
        eval_out (string): The path to store the predicted images in.

    Returns: None

    """
    if not os.path.exists(eval_out):
        os.makedirs(eval_out)

    patches_out = os.path.join(eval_out, "pred_patches")

    if not os.path.exists(patches_out):
        os.makedirs(patches_out)

    if isinstance(model, torch.nn.DataParallel):
        iters = model.module.iters

    else:
        iters = model.iters

    model.eval()

    with tqdm(total=len(dataloader), desc=f'Predicting...',
              unit='batch') as pbar:

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                if torch.cuda.is_available():
                    imgs = data['image'].cuda()

                else:
                    imgs = data['image']

                output, outputs = model(imgs)

                pred_bool = (output > 0.5).float()

                for j in range(imgs.shape[0]):
                    pred_name = os.path.split(data['img_name'][j])[-1].replace(".png",
                                                                   "_predicted.png")
                    pred_name = os.path.join(patches_out, pred_name)
                    output_bool = (output[j] > 0.5).float()

                    save_image(output_bool, pred_name)
                pbar.update(1)

    full_masks_pred = os.path.join(patches_out, "assembled_masks")

    if not os.path.exists(full_masks_pred):
        os.makedirs(full_masks_pred)

    patch_operations.reassemble_images(patches_out, full_masks_pred)


def main(args):

    if args.gcloud_checkpoint_path:
        data.get_model_from_gcloud(args.gcloud_checkpoint_path,
                              args.checkpoint_path)

    torch.manual_seed(args.seed)

    if args.gcloud:
        path = data.download_from_gcloud(args.gcloud_set_name,
                                            args.storage_bucket_id)
    else:
        path = args.dataset_path

    model, epoch = _load_model(args.checkpoint_path)

    if args.patch_set:
        dataloader = data.load_patch_set(path, args.seed, args.batch_size)

        predict_patches(dataloader, model, args.eval_out)

    else:
        dataloader = data.load_full_set(path, args.seed, args.batch_size)

        predict_full(dataloader, model, args.eval_out)



if __name__ == "__main__":
    args = get_args()
    main(args)
