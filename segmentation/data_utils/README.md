# data_utils
This folder contains various scripts for managing data related to segmentation.

## `balance_patches.py`
This script can be used to remove completely empty patches from a patch dataset to reduce class imbalance.

```
usage: balance_patches.py [-h] [-p KEEP_PROB] in_dir out_dir

Remove empty patches.

positional arguments:
  in_dir                The path to the images
  out_dir               The path to store the new dataset

optional arguments:
  -h, --help            show this help message and exit
  -p KEEP_PROB, --keep_prob KEEP_PROB
                        The probability to keep an empty patch
```

## `patch_generator.py`
This script is used to generate a patch dataset for training and evaluating the segmentation network.  It works by cropping the extent of the segmentation ground truth, and then randomly selecting patches of a given size from within that crop.  These patches are output to a chosen directory.

```
usage: patch_generator.py [-h] [--train] [--test] [-p PERCENT_PATCHES] [-c CROP_PADDING] [-s PATCH_SHAPE [PATCH_SHAPE ...]] csv_file out_dir

Create a set of patches from the paths given in csv_file.Patches willbe stored in out_dir.

positional arguments:
  csv_file              The path to the csv_file containing mask paths to be turned into patches.
  out_dir               The path to the output directory. If the directory contains an images and masks dir, patches will be stored in their respective directories, otherwise all patches will be stored
                        together in the out_dir.

optional arguments:
  -h, --help            show this help message and exit
  --train               Use this flag togenerate patches for training sets.
  --test                Use this flag togenerate patches for testing and validation sets.
  -p PERCENT_PATCHES, --percent_patches PERCENT_PATCHES
                        Spceifies the pecentage of the ratio of patches to make relative to the total possible number of patches for training sets. Basically, we need a specific number of patches to create,
                        but after cropping each image is a different size, so we can't always say make 100 patches. This is a floating point number that says, make x percent of the total possible number of
                        patches we could make.Default is 0.75
  -c CROP_PADDING, --crop_padding CROP_PADDING
                        The number of pixels to add to each side of the image when cropping around the foreground. Default is 256
  -s PATCH_SHAPE [PATCH_SHAPE ...], --patch_shape PATCH_SHAPE [PATCH_SHAPE ...]
                        The x, y dimensions of each patch. Default is (256, 256)
```

## `plot_loss.py`
This script is used to plot the loss values oat each iteration through the segmentation network.

```
usage: plot_loss.py [-h] train_file test_file out_file

Plot Training Loss

positional arguments:
  train_file  The path to the csv file contining training losses
  test_file   The path to the csv file containing test losses.
  out_file    The path to the output file.

optional arguments:
  -h, --help  show this help message and exit
```

## `train_splitter.py`

This script is used to create a train, test, validation split from a set of images and masks.


