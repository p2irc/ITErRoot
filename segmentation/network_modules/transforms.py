"""
File Name: transforms.py

Authors: Kyle Seidenthal

Date: 24-09-2020

Description: Callable transforms for data augmentation.

    These should be used with torchvision.transforms.Compose([Transform(args)])
    as the transform argument to the dataset.

"""
import torch
import numpy as np
from skimage.transform import rotate
import torchvision


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        if "mask" not in sample.keys():
            image = sample['image']

            to_tens = torchvision.transforms.ToTensor()

            image = to_tens(image)

            return {'image': image}
        else:
            image, mask = sample['image'], sample['mask']

            to_tens = torchvision.transforms.ToTensor()

            image = to_tens(image)
            mask = to_tens(mask)

            return {'image': image,
                    'mask': mask}


class Normalize(object):
    """ Normalization for dict format """

    def __init__(self, mean, std):

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, sample):

        if "mask" not in sample.keys():
            image = sample['image']
            image = image.float()

            norm = torchvision.transforms.Normalize(self.mean, self.std)

            image = image.float()
            image = norm(image)

            return {'image': image}
        else:
            image, mask = sample['image'], sample['mask']
            image = image.float()
            norm = torchvision.transforms.Normalize(self.mean, self.std)

            image = image.float()
            image = norm(image)

            return {'image': image,
                    'mask': mask}


class RandomHorizontalFlip(object):

    """Randomly flip the image horizontally"""

    def __init__(self, prob=0.5):
        """Create a new Transform.

        Kwargs:
            prob (float): The probability to flip. (Default 0.5)


        """
        if prob > 1 or prob < 0:
            raise ValueError("Probability values must be between 0 and 1.")

        self._prob = prob

    def __call__(self, sample):

        flip_chance = np.random.uniform(0, 1)

        if "mask" not in sample.keys():
            image = sample['image']

            if flip_chance <= self._prob:
                image = np.fliplr(image)

            return {'image': image}

        else:
            image, mask = sample['image'], sample['mask']


            if flip_chance <= self._prob:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            return {'image': image,
                    'mask': mask}


class RandomVerticalFlip(object):

    """Randomly flip the image vertically"""

    def __init__(self, prob=0.5):
        """Create a new Transform.

        Kwargs:
            prob (float): The probability to flip. (Default 0.5)


        """
        if prob > 1 or prob < 0:
            raise ValueError("Probability values must be between 0 and 1.")

        self._prob = prob

    def __call__(self, sample):

        flip_chance = np.random.uniform(0, 1)

        if "mask" not in sample.keys():
            image = sample['image']

            if flip_chance <= self._prob:
                image = np.flipud(image)

            return {'image': image}


        else:
            image, mask = sample['image'], sample['mask']

            if flip_chance <= self._prob:
                image = np.flipud(image)
                mask = np.flipud(mask)

            return {'image': image,
                    'mask': mask}


class RandomFlip(object):

    """Randomly flip the image vertically"""

    def __init__(self, prob=0.5, no_flip=0.0):
        """Create a new Transform.

        Kwargs:
            prob (float): The probability to flip horizontally. (Default 0.5)
                          The probability to flip vertically will be 1-prob,
                          unless no_flip is specified.
            no_flip (float): The probability to not flip at all.


        """
        if prob > 1 or prob < 0:
            raise ValueError("Probability values must be between 0 and 1.")

        self._prob = prob
        self._no_flip = no_flip

    def __call__(self, sample):

        flip_chance = np.random.uniform(0, 1)

        if "mask" not in sample.keys():
            image = sample['image']

            if flip_chance > self._no_flip:

                flip_chance = np.random.uniform(0, 1)

                if flip_chance <= self._prob:
                    image = np.fliplr(image)
                else:
                    image = np.flipud(image)

            return {'image': image}



        else:
            image, mask = sample['image'], sample['mask']


            if flip_chance > self._no_flip:

                flip_chance = np.random.uniform(0, 1)

                if flip_chance <= self._prob:
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)
                else:
                    image = np.flipud(image)
                    mask = np.flipud(mask)

            return {'image': image,
                    'mask': mask}


class RandomRotation(object):

    """    A random rotation augmentation. """

    def     __init__(self, max_degrees=90, min_degrees=0):
        """Create a random rotation Transform.

        Kwargs:
            max_degrees (int): The most amount of rotation, in degreees.
            min_degrees (int): The least amount of rotation, in degrees.

        Returns: A rotation transform.

        """

        self._max_degrees = max_degrees
        self._min_degrees = min_degrees

    def __call__(self, sample):

        angle = np.random.randint(self._min_degrees, self._max_degrees + 1)

        if "mask" not in sample.keys():
            image = sample['image']
            image = rotate(image, angle)

            return {'image': image}

        else:
            image, mask = sample['image'], sample['mask']

            image = rotate(image, angle)
            mask = rotate(mask, angle)

            return {'image': image,
                    'mask': mask}
