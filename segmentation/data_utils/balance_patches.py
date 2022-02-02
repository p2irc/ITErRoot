"""
File Name: balance_patches.py

Authors: Kyle Seidenthal

Date: 12-01-2021

Description: A script to remove completely empty patches in a balanced way.

"""

import os
import numpy as np
import skimage.io as io
import argparse

from shutil import copyfile

def check_empty(img):
    """Check if the given image is empty.

    Args:
        img (ndarray): The image to check.

    Returns: True if the image is all 0, False otherwise.

    """
    return not img.any()


def process(in_dir, out_dir, keep_prob=0.10):

    imgs_dir = os.path.join(in_dir, "images")
    masks_dir = os.path.join(in_dir, "masks")

    imgs_out_dir = os.path.join(out_dir, "images")
    masks_out_dir = os.path.join(out_dir, "masks")

    mask_paths = os.listdir(masks_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(imgs_out_dir)
        os.makedirs(masks_out_dir)

    total_imgs_before = 0
    total_imgs_after = 0
    total_empty_before = 0
    total_empty_after = 0

    for mask_path in mask_paths:
        total_imgs_before += 1

        copy = True

        full_mask_path = os.path.join(masks_dir, mask_path)

        img = io.imread(full_mask_path)

        if check_empty(img):
            total_empty_before += 1

            keep_chance = np.random.uniform(0, 1)

            if keep_chance < keep_prob:

                copy = True
                total_empty_after += 1

            else:
                copy = False


        mask_name = os.path.splitext(mask_path)[0]
        img_name = mask_name.split("_mask")[0] + ".png"

        full_img_path = os.path.join(imgs_dir, img_name)

        full_mask_out = os.path.join(masks_out_dir, mask_path)
        full_img_out = os.path.join(imgs_out_dir, img_name)

        if not os.path.exists(full_img_path):
            if check_empty(img):
                total_empty_after -= 1
            continue

        if copy:
            total_imgs_after += 1
            copyfile(full_mask_path, full_mask_out)
            copyfile(full_img_path, full_img_out)


    print("Before: Total {} Empty {}".format(total_imgs_before,
                                             total_empty_before))
    print("After: Total{} Empty {}".format(total_imgs_after, total_empty_after))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Remove empty patches.')

    parser.add_argument("in_dir", help="The path to the images")

    parser.add_argument("out_dir", help="The path to store the new dataset")

    parser.add_argument("-p", "--keep_prob", type=float, help="The probability to keep an "
                        "empty patch", default=0.1)
    args = parser.parse_args()

    process(args.in_dir, args.out_dir, keep_prob=args.keep_prob)


