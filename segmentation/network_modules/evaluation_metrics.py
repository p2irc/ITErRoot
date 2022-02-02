"""
File Name: evaluation_metrics.py

Authors: Kyle Seidenthal

Date: 25-11-2020

Description: Evaluation Metrics.

"""

import torch

def F1_score(predicted, target):
    """The F1 score for the predicted segmentation.

    Args:
        predicted (tensor): The predicted segmentation.
        target (tensor): The ground truth segmentation.

    Returns: The F1 score.

    """
    # Flatten the images
    predicted = predicted.view(-1)
    target = target.view(-1)

    intersect = (predicted * target).sum()

    union = (predicted.sum() + target.sum() + 1e-5)

    dice = (2. * intersect + 1e-5) / union

    return dice.item()

    #TP, FP, TN, FN = _get_confusion(predicted, target)

    #if TP == 0 and FP == 0 and FN == 0:
    #    return 0.0

    #return (2 * TP) / ((2 * TP) + FP + FN)


def IoU_score(predicted, target):
    """The IoU score for the predicted segmentation.

    Args:
        predicted (tensor): The predicted segmentation.
        target (tensor): The ground truth segmentation.

    Returns: The IoU score.

    """

    predicted = predicted.view(-1)
    target = target.view(-1)

    intersect = (predicted * target).sum()
    union = (predicted.sum() + target.sum() + 1e-5)


    return ((intersect + 1e-5) / union).item()
    #TP, FP, TN, FN = _get_confusion(predicted, target)

    #if TP == 0 and FP == 0 and FN == 0:
    #    return 0.0

    #return TP / (TP + FP + FN)


def pixel_wise_acc(predicted, target):
    """The pixel wise classification accuracy.

    Args:
        predicted (tensor): The predicted segmentation.
        target (tensor): The ground truth segmentation.

    Returns: The pixel wise classification accuracy.

    """

    predicted = predicted.view(-1)
    target = target.view(-1)

    total = predicted.size(0)


    correct = predicted.eq(target).sum()

    return (correct.item() / total)

    #TP, FP, TN, FN = _get_confusion(predicted, target)
    #return (TP + TN) / (TP + TN + FP + FN)


def get_confusion(predicted, target):
    """Get the confusion matrix for the batch.

    Args:
        predicted ({% TYPE %}): {% TODO %}
        target ({% TYPE %}): {% TODO %}

    Returns: {% TODO %}

    """
    predicted = predicted.view(-1).cpu()
    target = target.view(-1).cpu()

    from sklearn.metrics import confusion_matrix

    return confusion_matrix(target.numpy(), predicted.numpy(), labels=[0, 1])
