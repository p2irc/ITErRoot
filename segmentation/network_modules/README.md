# network_modules

Modules related to the structure of the segmentation network.

## `blocks.py`
This module contains definitions of the basic building blocks that make up the segmentation network.  It includes:

- BasicBlock:
	- A simple residual block consisting of Convolutional and Batchnorm layers
- Down:
	- A downward convolutional block that uses a max-pooling layer with a BasicBlock
- Up:
	- An upward convolutional block that uses an upward convolutional layer with a BasicBlock
- InputBlock:
	- A block that acts as the input layer to the network
- OutputBLock:
	- A block that acts as the ouput to a binary segmentation probability map


## `dataset.py`
This module defines the dataloader to be used for loading patch images for the segmentation network.  It also has functionality for downloading the dataset from GCloud and creating the appropriate dataloader.  Normalization data and transforms are applied here.

## `early_stopping.py`
This module defines mechanisms for using early stopping.

## `evaluation_metrics.py`
Ths module contains definitions of evaluation metrics for measuring model results.
Available metrics are:
- Sensitivity
- Specificity
- F1 (Dice)
- IoU
- Pixel Wise Accuracy

## `loss.py` 
Definitions of loss functions for use with the segmentation network.
Available functions are:
- CombinedDLCrossent
	- Dice Loss + Binary Crossentropy
- CombinedWDLCrossent
	- (a * Dice Loss) + (b * Binary Crossentropy)
- CombinedIoUCrossent
	- IoU Loss + Binary Crossentropy
- Dice Loss
- IoU Loss

## `network_structure.py`
This module contains the pieces used to build the iterative network structure.

## `ITErRoot.py`
The main definition of the network.  This is what you use to train and evaluate.

## `transforms.py`
Definitions of classes for data augmentation.

## `patch_operations.py`
This contains operations to patchify and un-patchify an image for prediction of full sized image sets.
