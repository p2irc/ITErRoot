"""
File Name: train_splitter.py

Authors: Kyle Seidenthal

Date: 01-10-2020

Description: Script to split train set into train and test sets.

"""
import os
import random
import csv

TRAIN_DIR = "/path/to/Train/masks"
TEST_DIR = "/path/to/Test/masks"
TEST_PERCENT = 0.15

# Get list of images in Train
train_images = os.listdir(TRAIN_DIR)

# Split into train and test
num_test = int(round(TEST_PERCENT * len(train_images)))

random.shuffle(train_images)

train_set = train_images[num_test:]
test_set = train_images[:num_test]


# Write train images to 'train_patches.csv'
with open('train_patches.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    for image in train_set:
        name = os.path.join(TRAIN_DIR, image)
        writer.writerow([name])

# Write test images to test_patches.csv
with open('test_patches.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    for image in test_set:
        name = os.path.join(TRAIN_DIR, image)
        writer.writerow([name])


# Create 'valid_patches.csv' from test dir, yes, I really do mean test
# it's because I did a dumb smart thing a while back
with open('valid_patches.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    for image in os.listdir(TEST_DIR):
        name = os.path.join(TEST_DIR, image)
        writer.writerow([name])

# These csv files will be used by the patch generation script
