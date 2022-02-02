"""
File Name: iou.py

Authors: Kyle Seidenthal

Date: 30-06-2021

Description: A script for computing evaluation metrics on a set of
segmentations.

"""
import numpy as np
import argparse
import os
import json
import math

from skimage import io
from sklearn.metrics import (f1_score, jaccard_score, accuracy_score,
                            confusion_matrix)
from skimage.util import img_as_float, img_as_bool
from skimage.color import rgb2gray
from skimage.transform import rotate

from tqdm import tqdm

def main(args):

    pred_path = args.pred_path
    gt_path = args.gt_path

    pred_imgs = os.listdir(pred_path)
    gt_img = os.listdir(gt_path)

    metrics = {}

    metrics["IoU"] = {"values": np.array([]),
                      "mean": -1,
                      "max": -1,
                      "min": -1,
                      "std": -1}

    metrics["dice"] = {"values": np.array([]),
                      "mean": -1,
                      "max": -1,
                      "min": -1,
                      "std": -1}

    metrics["acc"] = {"values": np.array([]),
                      "mean": -1,
                      "max": -1,
                      "min": -1,
                      "std": -1}

    metrics["sens"] = {"values": np.array([]),
                      "mean": -1,
                      "max": -1,
                      "min": -1,
                      "std": -1}

    metrics["spec"] = {"values": np.array([]),
                      "mean": -1,
                      "max": -1,
                      "min": -1,
                      "std": -1}


    for pred in tqdm(pred_imgs):

        pred_img = io.imread(os.path.join(pred_path, pred))
        pred_img = img_as_bool(rgb2gray(pred_img))

        gt_name = os.path.join(gt_path, pred.replace("predicted", "mask"))

        if ".jpg" in gt_name:
            gt_name = gt_name.replace(".jpg", ".png")

        gt_img = io.imread(os.path.join(gt_name))
        gt_img = img_as_bool(rgb2gray(gt_img))

        # Some of the tiff tags messed up rotation of the GT, specifically in
        # the case of the Full Image Cucumber set.  Here we rotate the GT 90
        # degrees clockwise if necessary (270 degrees counter clockwise)
        if gt_img.shape[0] > pred_img.shape[0] and gt_img.shape[1] < pred_img.shape[1]:
            gt_img = img_as_bool(rotate(gt_img, 270, resize=True,
                                        preserve_range=True))

        # Crop to size
        pred_img = pred_img[:gt_img.shape[0], :gt_img.shape[1]]

        w, h = pred_img.shape

        pred_img = pred_img.reshape(-1)
        gt_img = gt_img.reshape(-1)



        IoU = jaccard_score(gt_img, pred_img, zero_division=1.0)
        dice = f1_score(gt_img, pred_img, zero_division=1.0)
        acc = accuracy_score(gt_img, pred_img)

        tn, fp, fn, tp = confusion_matrix(gt_img, pred_img, labels=[0, 1]).ravel()


        if (tp + fn) == 0:
            sensitivity = 1.0
        else:
            sensitivity = tp / (tp + fn)

        if (tn + fp) == 0:
            specificity = 1.0
        else:
            specificity = tn / (tn + fp)


        if math.isnan(IoU) or math.isnan(dice) or math.isnan(acc) or math.isnan(sensitivity) or math.isnan(specificity):
            print("Found NaN")
            print(os.path.join(pred_path, pred))
            print(IoU, dice, acc, sensitivity, specificity)
            import sys
            sys.exit(1)

        metrics["IoU"]["values"] = np.append(metrics["IoU"]["values"], IoU)
        metrics["dice"]["values"] = np.append(metrics["dice"]["values"], dice)
        metrics["acc"]["values"] = np.append(metrics["acc"]["values"], acc)
        metrics["sens"]["values"] = np.append(metrics["sens"]["values"], sensitivity)
        metrics["spec"]["values"] = np.append(metrics["spec"]["values"], specificity)


    for k in metrics.keys():
        metrics[k]["mean"] = np.mean(metrics[k]["values"])
        metrics[k]["max"] = np.max(metrics[k]["values"])
        metrics[k]["min"] = np.min(metrics[k]["values"])
        metrics[k]["std"] = np.std(metrics[k]["values"])
        metrics[k]["values"] = metrics[k]["values"].tolist()

    with open(args.out_path, 'w') as jsonfile:
        json.dump(metrics, jsonfile)


def get_args():

    parser = argparse.ArgumentParser("Get evaluation metrics for images.")

    parser.add_argument("pred_path",
                        type=str,
                        help="The path to the predicted segmentations.")

    parser.add_argument("gt_path",
                        type=str,
                        help="The path to the ground truth segmentations.")

    parser.add_argument("out_path",
                        type=str,
                        help="The path to store the metrics json file.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    main(args)

