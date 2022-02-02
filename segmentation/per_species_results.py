"""
File Name: _per_species_results.py

Authors: {% <AUTHOR> %}

Date: 12-07-2021

Description: {% <DESCRIPTION> %}

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

def get_res(pred_img, gt_img):
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

    return dice, IoU, acc, sensitivity, specificity


def main(args):

    pred_path = args.pred_path
    gt_path = args.gt_path

    pred_imgs = os.listdir(pred_path)
    gt_img = os.listdir(gt_path)

    cucumber_res = {"dice": np.array([]),
                    "acc": np.array([]),
                    "IoU": np.array([]),
                    "sens": np.array([]),
                    "spec": np.array([])}

    canola_res = {"dice": np.array([]),
                    "acc": np.array([]),
                    "IoU": np.array([]),
                    "sens": np.array([]),
                    "spec": np.array([])}

    soy_res = {"dice": np.array([]),
                    "acc": np.array([]),
                    "IoU": np.array([]),
                    "sens": np.array([]),
                    "spec": np.array([])}

    wheat_res = {"dice": np.array([]),
                    "acc": np.array([]),
                    "IoU": np.array([]),
                    "sens": np.array([]),
                    "spec": np.array([])}


    for pred in tqdm(pred_imgs):

        species = int(pred.split("-")[1])
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

        dice, IoU, acc, sens, spec = get_res(pred_img, gt_img)

        if species == 0:
            cucumber_res["dice"] = np.append(cucumber_res["dice"], dice)
            cucumber_res["acc"] = np.append(cucumber_res["acc"], acc)
            cucumber_res["IoU"] = np.append(cucumber_res["IoU"], IoU)
            cucumber_res["sens"] = np.append(cucumber_res["sens"], sens)
            cucumber_res["spec"] = np.append(cucumber_res["spec"], spec)

        elif species == 1:
            canola_res["dice"] = np.append(canola_res["dice"], dice)
            canola_res["acc"] = np.append(canola_res["acc"], acc)
            canola_res["IoU"] = np.append(canola_res["IoU"], IoU)
            canola_res["sens"] = np.append(canola_res["sens"], sens)
            canola_res["spec"] = np.append(canola_res["spec"], spec)


        elif species == 2:
            soy_res["dice"] = np.append(soy_res["dice"], dice)
            soy_res["acc"] = np.append(soy_res["acc"], acc)
            soy_res["IoU"] = np.append(soy_res["IoU"], IoU)
            soy_res["sens"] = np.append(soy_res["sens"], sens)
            soy_res["spec"] = np.append(soy_res["spec"], spec)

        elif species == 3:
            wheat_res["dice"] = np.append(wheat_res["dice"], dice)
            wheat_res["acc"] = np.append(wheat_res["acc"], acc)
            wheat_res["IoU"] = np.append(wheat_res["IoU"], IoU)
            wheat_res["sens"] = np.append(wheat_res["sens"], sens)
            wheat_res["spec"] = np.append(wheat_res["spec"], spec)

        else:
            print("Invalid species")

    # Cucumber
    cuc_dice = np.mean(cucumber_res["dice"])
    cuc_dice_std = np.std(cucumber_res["dice"])

    cuc_acc = np.mean(cucumber_res["acc"])
    cuc_acc_std = np.std(cucumber_res["acc"])

    cuc_IoU = np.mean(cucumber_res["IoU"])
    cuc_IoU_std = np.std(cucumber_res["IoU"])

    cuc_sens = np.mean(cucumber_res["sens"])
    cuc_sens_std = np.std(cucumber_res["sens"])

    cuc_spec = np.mean(cucumber_res["spec"])
    cuc_spec_std = np.std(cucumber_res["spec"])

    #Canola
    canola_dice = np.mean(canola_res["dice"])
    canola_dice_std = np.std(canola_res["dice"])

    canola_acc = np.mean(canola_res["acc"])
    canola_acc_std = np.std(canola_res["acc"])

    canola_IoU = np.mean(canola_res["IoU"])
    canola_IoU_std = np.std(canola_res["IoU"])

    canola_sens = np.mean(canola_res["sens"])
    canola_sens_std = np.std(canola_res["sens"])

    canola_spec = np.mean(canola_res["spec"])
    canola_spec_std = np.std(canola_res["spec"])

    # Soy
    soy_dice = np.mean(soy_res["dice"])
    soy_dice_std = np.std(soy_res["dice"])

    soy_acc = np.mean(soy_res["acc"])
    soy_acc_std = np.std(soy_res["acc"])

    soy_IoU = np.mean(soy_res["IoU"])
    soy_IoU_std = np.std(soy_res["IoU"])

    soy_sens = np.mean(soy_res["sens"])
    soy_sens_std = np.std(soy_res["sens"])

    soy_spec = np.mean(soy_res["spec"])
    soy_spec_std = np.std(soy_res["spec"])

    # Wheat
    wheat_dice = np.mean(wheat_res["dice"])
    wheat_dice_std = np.std(wheat_res["dice"])

    wheat_acc = np.mean(wheat_res["acc"])
    wheat_acc_std = np.std(wheat_res["acc"])

    wheat_IoU = np.mean(wheat_res["IoU"])
    wheat_IoU_std = np.std(wheat_res["IoU"])

    wheat_sens = np.mean(wheat_res["sens"])
    wheat_sens_std = np.std(wheat_res["sens"])

    wheat_spec = np.mean(wheat_res["spec"])
    wheat_spec_std = np.std(wheat_res["spec"])


    out_data = {}

    canola_data =  {"dice": canola_dice,
                    "dice_std" : canola_dice_std,
                    "acc": canola_acc,
                    "acc_std": canola_acc,
                    "IoU": canola_IoU,
                    "IoU_std": canola_IoU_std,
                    "sens": canola_sens,
                    "sens_std": canola_sens_std,
                    "spec": canola_spec,
                    "spec_std": canola_spec_std
                    }

    wheat_data =  {"dice": wheat_dice,
                    "dice_std" : wheat_dice_std,
                    "acc": wheat_acc,
                    "acc_std": wheat_acc,
                    "IoU": wheat_IoU,
                    "IoU_std": wheat_IoU_std,
                    "sens": wheat_sens,
                    "sens_std": wheat_sens_std,
                    "spec": wheat_spec,
                    "spec_std": wheat_spec_std
                    }

    soy_data =  {"dice": soy_dice,
                    "dice_std" : soy_dice_std,
                    "acc": soy_acc,
                    "acc_std": soy_acc,
                    "IoU": soy_IoU,
                    "IoU_std": soy_IoU_std,
                    "sens": soy_sens,
                    "sens_std": soy_sens_std,
                    "spec": soy_spec,
                    "spec_std": soy_spec_std
                    }

    cuc_data =  {"dice": cuc_dice,
                    "dice_std" : cuc_dice_std,
                    "acc": cuc_acc,
                    "acc_std": cuc_acc,
                    "IoU": cuc_IoU,
                    "IoU_std": cuc_IoU_std,
                    "sens": cuc_sens,
                    "sens_std": cuc_sens_std,
                    "spec": cuc_spec,
                    "spec_std": cuc_spec_std
                    }


    out_data["canola"] = canola_data
    out_data["soy"] = soy_data
    out_data["wheat"] = wheat_data
    out_data["cucumber"] = cuc_data
    import json

    with open(args.out_path, 'w') as jsonfile:
        json.dump(out_data, jsonfile)



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

