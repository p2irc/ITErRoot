"""
File Name: patch_operations.py

Authors: Kyle Seidentha

Date: 24-02-2021

Description: Operations to split and reconstruct image patches.

"""
import os
import skimage.io as io
import numpy as np

from tqdm import tqdm
from skimage.util.shape import view_as_windows

import multiprocessing

from skimage.util import img_as_bool, img_as_uint, img_as_float
from skimage.color import rgb2gray

from PIL import Image, ExifTags
from PIL.TiffTags import TAGS

def fix_orientation(image_path, mask_path):

    if os.path.splitext(image_path)[-1] != ".tif":
        return
    TIFFTAG_ORIENTATION = 274
    img = Image.open(image_path)
    gt = Image.open(mask_path)

    #for orientation in ExifTags.TAGS.keys():
    #    if ExifTags.TAGS[orientation] == 'orientation':
    #        break

    try:
        img_orientation = img.tag[TIFFTAG_ORIENTATION]

        if img_orientation[0] == 3:
            gt = gt.rotate(180, expand=True)
        elif img_orientation[0] == 6:
            gt = gt.rotate(270, expand=True)
        elif img_orientation[0] == 8:
            gt = gt.rotate(90, expand=True)
    except:
        pass

    print("Fixing orientation for {}".format(image_path))
    gt.save(mask_path)
    gt.close()


def create_patches(input_dir, output_dir, shape=(256, 256), overlap=128):

    images = os.listdir(input_dir)

    pool = multiprocessing.Pool()

    list(tqdm(pool.imap(_create_patches_help, [(input_dir, image_name,
                                                output_dir, shape, overlap) for
                                               image_name in images]),
              total=len(images), desc="Patchifying Images"))


def _create_patches_help(args):
    input_dir, image_name, output_dir, shape, overlap = args

    image_path = os.path.join(input_dir, image_name)

    if not os.path.isdir(image_path):
        patchify_image(image_path, output_dir, shape, overlap)


def patchify_image(image_path, out_path, shape=(256, 256), overlap=128):

    img = io.imread(image_path, plugin='pil')

    image_name = os.path.splitext(os.path.split(image_path)[-1])[0]

    if len(img.shape) == 3:
        # Remove alpha channel
        if img.shape[2] == 4:
            img = img[:, :, :3]

        img_win = view_as_windows(img, (shape[0], shape[1], 3), step=128)
    else:
        img_win = view_as_windows(img, (shape[0], shape[1]), step=128)


    for i in range(img_win.shape[0]):
        for j in range(img_win.shape[1]):

            if len(img.shape) == 3:
                patch_img = img_win[i, j][0]
                if patch_img.shape != (256, 256, 3):
                    pad_x = (0, 256 - patch_img.shape[0])
                    pad_y = (0, 256 - patch_img.shape[1])

                    patch_img = np.pad(img, (pad_x, pad_y, (0, 0)), 'constant',
                                       constant_values=(0, 0))

                patch_name = image_name + "_" + str(i) + "_" + str(j) + ".png"

            else:
                patch_img = img_win[i, j]
                if patch_img.shape != (256, 256):
                    pad_x = (0, 256 - patch_img.shape[0])
                    pad_y = (0, 256 - patch_img.shape[1])

                    patch_img = np.pad(img, (pad_x, pad_y), 'constant',
                                       constant_values=(0, 0))
                image_name = image_name.split("_mask")[0]
                patch_name = image_name + "_" + str(i) + "_" + str(j) + "_mask.png"

            out_name = os.path.join(out_path, patch_name)

            io.imsave(out_name, patch_img, check_contrast=False)

def reassemble_images(input_dir, output_dir, overlap=128):

    images = os.listdir(input_dir)

    image_groups = {}

    for image_name in tqdm(images, desc="Grouping Patches"):
        if os.path.isdir(os.path.join(input_dir, image_name)):
            continue

        img_id = "_".join(image_name.split("_")[:-3])
        i, j, _ = image_name.split("_")[-3:]
        #img_id, i, j, _ = image_name.split("_")

        i = int(i)
        j = int(j)

        if img_id not in image_groups.keys():
            image_groups[img_id] = {image_name: image_name,
                                     "max_i": i,
                                     "max_j": j}
        else:
            image_groups[img_id][image_name] = image_name

            if i > image_groups[img_id]["max_i"]:
                image_groups[img_id]["max_i"] = i

            if j > image_groups[img_id]["max_j"]:
                image_groups[img_id]["max_j"] = j


    pool = multiprocessing.Pool()

    list(tqdm(pool.imap(_assemble_image_help, [(image_groups, group_key, input_dir,
                                                output_dir, overlap) for
                                               group_key in image_groups.keys()]),
              total=len(image_groups.keys()), desc="Assembling Images"))

def _assemble_image_help(args):
    groups, group_key, input_dir, output_dir, overlap = args
    _assemble_image(groups, group_key, input_dir, output_dir, overlap)

def _assemble_image(groups, group_key, input_dir, output_dir, overlap):
    group = groups[group_key]

    max_i = group["max_i"]
    max_j = group["max_j"]

    shape_x = ((max_i + 1) * overlap) + 256
    shape_y = ((max_j + 1) * overlap) + 256

    full_img = np.zeros((shape_x, shape_y), dtype=np.float)

    for i in range(max_i + 1):
        for j in range(max_j + 1):
            patch_name = group_key + "_" + str(i) + "_" + str(j) + "_predicted.png"#"_mask.png"
            patch_name = os.path.join(input_dir, patch_name)
            patch = img_as_float(rgb2gray(io.imread(patch_name)))

            starti = i * overlap
            endi = starti + patch.shape[0]
            startj = j * overlap
            endj = startj + patch.shape[1]

            full_img[starti:endi, startj:endj] += patch

            if i == 0 or i == max_i:
                if j != 0 and j != max_j:
                    full_img[starti:endi, startj:endj] += 1

            if j == 0 or j == max_j:
                if i !=0 and i != max_i:
                    full_img[starti:(starti + overlap), startj:endj] += 1



    full_img = (full_img > 2)


    output_name = group_key + "_predicted.png"
    output_name = os.path.join(output_dir, output_name)
    io.imsave(output_name, img_as_uint(full_img), check_contrast=False)
