"""
File Name: patch_generator.py

Authors: Kyle Seidenthal

Date: 05-10-2020

Description: Generate a patch dataset for training and testing.

"""

import skimage.io as io
import numpy as np
import os
import csv
import argparse
import sys

from skimage.util.shape import view_as_blocks
from math import floor
from tqdm import tqdm


def create_train_patch(img_path, mask_path, percent_patches, out_dir,
                       crop_padding, patch_shape=(256, 256)):
    """Create patches from the training image by:
        1. Cropping the image to remove extra background.
        2. Selecting n random patches from the cropped image.

    Args:
        img_path (string): Path to the image to make patches from
        mask_path (string): Path to the mask for the image
        percent_patches (float): The percentage of the image to cover with
                                 patches.  This makes sure we cover a majority
                                 of the image proportional to its size.
        out_dir (string): Path to the output directory for training data.
                          If an images/ and masks/ directory are available,
                          patches will be sorted.

        crop_padding (int): Number of pixels padding to use when cropping
        patch_shape (tuple): (x, y) size of each patch

    Returns: None, the patches are saved to out_dir

    """
    dirs = os.listdir(out_dir)

    out_dir_img = None
    out_dir_mask = None

    for d in dirs:
        if os.path.isdir(os.path.join(out_dir, d)):
            if "images" in d:
                out_dir_img = os.path.join(out_dir, d)
            elif "masks" in d:
                out_dir_mask = os.path.join(out_dir, d)

    mask = io.imread(mask_path)
    img = io.imread(img_path, plugin='pil')

    img_name = os.path.split(os.path.splitext(img_path)[0])[-1]

    cropped_img, cropped_mask = crop_and_pad(img, mask, crop_padding)

    num_patches_x = cropped_mask.shape[0] // patch_shape[0]
    num_patches_y = cropped_mask.shape[1] // patch_shape[1]

    num_patches = num_patches_x * num_patches_y
    num_patches = int(percent_patches * num_patches)

    for i in range(num_patches):
        x = cropped_mask.shape[0]
        y = cropped_mask.shape[1]

        while (x + patch_shape[0]) >= cropped_mask.shape[0]:
            x = np.random.randint(0, cropped_mask.shape[0])

        while (y + patch_shape[1]) >= cropped_mask.shape[1]:
            y = np.random.randint(0, cropped_mask.shape[1])

        patch_img = cropped_img[x:x+patch_shape[0], y:y+patch_shape[1]]
        patch_mask = cropped_mask[x:x+patch_shape[0], y:y+patch_shape[1]]

        out_name = img_name + "_" + str(i) + ".png"
        out_name_mask = img_name + "_" + str(i) + "_mask.png"

        if out_dir_img is None:
            out_name = os.path.join(out_dir, out_name)
        else:
            out_name = os.path.join(out_dir_img, out_name)

        if out_dir_mask is None:
            out_name_mask = os.path.join(out_dir, out_name_mask)
        else:
            out_name_mask = os.path.join(out_dir_mask, out_name_mask)
        try:
            io.imsave(out_name, patch_img, check_contrast=False)
            io.imsave(out_name_mask, patch_mask, check_contrast=False)
        except ValueError as e:
            print(e)
            print(out_name, patch_img.shape)


def create_test_patch(img_path, mask_path, out_dir, crop_padding,
                      patch_shape=(256, 256)):
    """Create patches from the test image by splitting the entire image into
       non-overlapping patches.

    Args:
        img_path (string): Path to the image to make patches from.
        mask_path (string): Path to the mask for the image.
        out_dir (string): Path to the output directory
        crop_padding (int): The amount of padding around the cropped
                            foreground.
        patch_shape (tuple): The shape (x, y) of the patches

    Returns: None, patches are saved to out_dir

    """

    dirs = os.listdir(out_dir)

    out_dir_img = None
    out_dir_mask = None

    for d in dirs:
        if os.path.isdir(os.path.join(out_dir, d)):
            if "images" in d:
                out_dir_img = os.path.join(out_dir, d)
            elif "masks" in d:
                out_dir_mask = os.path.join(out_dir, d)

    img_name = os.path.split(os.path.splitext(img_path)[0])[-1]

    img = io.imread(img_path, plugin='pil')
    mask = io.imread(mask_path)

    img, mask = crop_and_pad(img, mask, crop_padding)

    num_patches_x = floor(img.shape[0] / patch_shape[0])
    num_patches_y = floor(img.shape[1] / patch_shape[1])

    patch_excess_x = img.shape[0] - (patch_shape[0] * num_patches_x)
    patch_excess_y = img.shape[1] - (patch_shape[1] * num_patches_y)

    pad_x = (0, patch_shape[0] - patch_excess_x)
    pad_y = (0, patch_shape[1] - patch_excess_y)

    img = np.pad(img, (pad_x, pad_y, (0, 0)), 'constant',
                 constant_values=(0, 0))
    mask = np.pad(mask, (pad_x, pad_y), 'constant', constant_values=(0, 0))

    # Make the blocks
    img_block_shape = (patch_shape[0], patch_shape[1], img.shape[2])

    img_blocks = view_as_blocks(img, block_shape=img_block_shape)
    mask_blocks = view_as_blocks(mask, block_shape=patch_shape)

    patch_num = 0
    for i in range(img_blocks.shape[0]):
        for j in range(img_blocks.shape[1]):
            patch_img = img_blocks[i, j][0]
            patch_mask = mask_blocks[i, j]

            out_name = img_name + "_" + str(patch_num) + ".png"
            out_name_mask = img_name + "_" + str(patch_num) + "_mask.png"

            patch_num += 1

            if out_dir_img is None:
                out_name = os.path.join(out_dir, out_name)
            else:
                out_name = os.path.join(out_dir_img, out_name)

            if out_dir_mask is None:
                out_name_mask = os.path.join(out_dir, out_name_mask)
            else:
                out_name_mask = os.path.join(out_dir_mask, out_name_mask)

            io.imsave(out_name, patch_img, check_contrast=False)
            io.imsave(out_name_mask, patch_mask, check_contrast=False)


def crop_and_pad(img, mask, padding):
    """Crop the given image around the foreground.

    Args:
        img (numpy array): The image to crop and pad.
        mask (numpy array): The mask to use to crop with.
        padding (int): The amount of padding around the foreground to include.

    Returns: The cropped image and mask.

    """
    rows, columns = np.where(mask)

    min_row = min(rows) - padding
    max_row = max(rows) + padding

    if min_row < 0:
        min_row = 0

    if max_row > mask.shape[0]:
        max_row = mask.shape[0]

    min_col = min(columns) - padding
    max_col = max(columns) + padding

    if min_col < 0:
        min_col = 0

    if max_col > mask.shape[1]:
        max_col = mask.shape[1]

    cropped_mask = mask[min_row:max_row, min_col:max_col]
    cropped_img = img[min_row:max_row, min_col:max_col]

    return cropped_img, cropped_mask


def load_mask_paths(csv_file):
    """Load paths from the given csv file.

    Args:
        csv_file (string): The path to the csv file to read.

    Returns: A list of path names.

    """

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)

        paths = []

        for row in reader:
            paths.append(row[0])

    return paths


def get_image_path(mask_path):
    """Get the image path from the given mask path.

    Args:
        mask_path (string): The path to the mask.

    Returns: The path to the image.

    """
    parts = os.path.split(mask_path)

    file_name = parts[-1]
    mask_path = parts[0]

    image_dir = os.path.split(mask_path)[0]
    image_dir = os.path.join(image_dir, "images")

    image_name = os.path.splitext(file_name)[0]
    image_code = image_name.split("_")[0]

    image_tif = image_code + ".tif"

    out_path = os.path.join(image_dir, image_tif)

    return out_path


def run(csv_file, out_dir, train, percent_patches=0.75, crop_padding=256,
        patch_shape=(256, 256)):

    patch_shape = (patch_shape[0], patch_shape[1])

    images = load_mask_paths(csv_file)

    for mask_path in tqdm(images, desc="Patching images..."):
        img_path = get_image_path(mask_path)

        if train:
            create_train_patch(img_path, mask_path, percent_patches, out_dir,
                               crop_padding, patch_shape=patch_shape)
        else:
            create_test_patch(img_path, mask_path, out_dir, crop_padding,
                              patch_shape=patch_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of patches from"
                                     " the paths given in csv_file."
                                     "Patches will"
                                     "be stored in out_dir.")

    parser.add_argument("csv_file", help="The path to the csv_file containing"
                        " mask paths to be turned into patches.")

    parser.add_argument("out_dir", help="The path to the output directory.  If"
                        " the directory contains an images and masks dir, "
                        "patches will be stored in their respective "
                        "directories, otherwise all patches will be stored "
                        "together in the out_dir.")

    parser.add_argument("--train", action="store_true", help="Use this flag to"
                        "generate patches for training sets.")

    parser.add_argument("--test", action="store_true", help="Use this flag to"
                        "generate patches for testing and validation sets.")

    parser.add_argument("-p", "--percent_patches", type=float, help="Spceifies"
                        " the pecentage of the ratio of patches to make "
                        "relative to the total possible number of patches "
                        "for training sets.  Basically, we need a specific "
                        "number of patches to create, but after cropping "
                        "each image is a different size, so we can't always "
                        "say make 100 patches.  This is a floating point "
                        "number that says, make x percent of "
                        "the total possible number of patches we could make."
                        "Default is 0.75",
                        default=0.75)

    parser.add_argument("-c", "--crop_padding", type=int, help="The number of "
                        "pixels to add to each side of the image when cropping"
                        " around the foreground.  Default is 256",
                        default=256)

    parser.add_argument("-s", "--patch_shape", type=int, help="The x, y "
                        "dimensions of each patch.  Default is (256, 256)",
                        nargs="+",
                        default=(256, 256))

    args = parser.parse_args()

    # Error checking

    if not os.path.exists(args.csv_file):
        print("The path {} does not exist.".format(args.csv_file))
        sys.exit(1)

    if not os.path.exists(args.out_dir):
        print("The path {} does not exist.".format(args.out_dir))
        sys.exit(1)

    if not args.train and not args.test:
        print("One of --test or --train must be set.")
        sys.exit(1)

    if args.train and args.test:
        print("Only one of --test or --train can be set at a time.")
        sys.exit(1)

    if args.percent_patches > 1 or args.percent_patches <= 0:
        print("Percent patches must be between 0 and 1, and cannot be zero.")
        sys.exit(1)

    if args.crop_padding < 0:
        print("Crop padding cannot be negative.")
        sys.exit(1)

    if len(args.patch_shape) != 2:
        print("Patch shapes must be two dimensional (x, y)")
        sys.exit(1)

    if args.patch_shape[0] <= 0 or args.patch_shape[1] <= 0:
        print("Patch shapes must be positive integers.")
        sys.exit(1)

    # Call it
    if args.train:
        run(args.csv_file, args.out_dir, True,
            percent_patches=args.percent_patches,
            crop_padding=args.crop_padding, patch_shape=args.patch_shape)

    elif args.test:
        run(args.csv_file, args.out_dir, False,
            percent_patches=args.percent_patches,
            crop_padding=args.crop_padding, patch_shape=args.patch_shape)
