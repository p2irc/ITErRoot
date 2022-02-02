"""
File Name: train.py

Authors: Kyle Seidenthal

Date: 13-01-2021

Description: Training script for SproutNetSeg.  This is set up  to run with
Docker and Google Cloud Services.

"""

import torch
import json
import csv
import argparse
import hypertune

import os
import numpy as np
import tarfile
import network_modules.dataset as data

from network_modules.loss import (CombinedDLCrossent, DiceLoss, IoULoss,
                                  CombinedIoUCrossent, CombinedWDLCrossent)
from network_modules.ITErRoot import ITErRoot
from network_modules.dataset import ITErRootTrainingDataset
from network_modules.transforms import (RandomFlip, RandomRotation, ToTensor,
                                        Normalize)
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from network_modules.early_stopping import EarlyStopperBase, EarlyStopperSingle
from torchsummary import summary
from tqdm import tqdm
from network_modules import evaluation_metrics as em
from google.cloud import storage


def _set_devices(model, criterion):
    """Set the device for the model.

    Args:
        model (model): The model to set up.
        gpus (list): List of gpus to use.

    Returns: The model on the correct device.

    """

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)

        if isinstance(criterion, list):
            for i in range(len(criterion)):
                criterion[i] = criterion[i].cuda()
        else:
            criterion = criterion.cuda()
    else:
        print("Using CPU")
        model = model.cpu()

    return model, criterion


def _train_step(dataloader, model, optimizer, epoch, epochs, criterion,
                num_minibatch=1):
    """ Perform one training step.

    Args:
        dataloader (DataLoader): The dataloader for training data.
        model (model): The model to train.
        epoch (int): The current epoch.
        epochs (int): Total epochs.
        criterion (loss):  The loss function.
    KWargs:
        num_minibatch (int): The number of batches before updating gradients.

    Returns: The loss for this trianing step.

    """
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch +1}/{epochs}',
              unit='batch') as pbar:

        epoch_loss = 0.0

        iter_losses = [0.0 for opt in optimizer]

        model.train()

        for i, data in enumerate(dataloader, 0):
            stepped = False

            if torch.cuda.is_available():
                imgs, masks = (data['image'].cuda(),
                               data['mask'].cuda())
            else:
                imgs, masks = (data['image'],
                               data['mask'])

            output, outputs = model(imgs)

            j = 0
            loss_index = 0
            for out, opt in zip(outputs, optimizer):

                if isinstance(criterion, list):
                    loss = criterion[loss_index](out, masks)
                    loss_index += 1
                    loss_index = loss_index % len(criterion)
                else:
                    loss = criterion(out, masks)

                loss.backward()

                if (i + 1) % num_minibatch == 0:
                    opt.step()
                    opt.zero_grad()

                    stepped = True

                iter_losses[j] += loss.item()
                j += 1

            epoch_loss += loss.item()

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            pbar.update(1)

        if not stepped:
            for opt in optimizer:
                opt.step()

        epoch_loss /= len(dataloader)
    print("Iteration Losses: ", [x/len(dataloader) for x in iter_losses])
    return epoch_loss, [x/len(dataloader) for x in iter_losses]


def _eval_step(dataloader, model, criterion, epoch):
    """Perfom an evaluation step.

    Args:
        dataloader (DataLoader): The dataloader for testing data.
        model (model): The model to use.
        criterion (loss): Loss function.
        epoch (int): The current epoch.

    Returns: The evaluation loss.

    """
    epoch_loss = 0.0
    avg_dice = 0.0

    if isinstance(model, torch.nn.DataParallel):
        iters = model.module.iters
    else:
        iters = model.iters

    iter_losses = [0.0 for i in range(iters + 1)]

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):

            if torch.cuda.is_available():
                imgs, masks = (data['image'].cuda(),
                               data['mask'].cuda())
            else:
                imgs, masks = (data['image'],
                               data['mask'])

            output, outputs = model(imgs)

            i = 0
            loss_index = 0
            for out in outputs:

                if isinstance(criterion, list):
                    loss = criterion[loss_index](out, masks)
                    loss_index += 1
                    loss_index = loss_index % len(criterion)

                else:
                    loss = criterion(out, masks)

                iter_losses[i] += loss.item()
                i += 1

            epoch_loss += loss.item()

            dice = em.F1_score(output, masks)

            avg_dice += dice

    epoch_loss /= len(dataloader)
    avg_dice /= len(dataloader)

    return epoch_loss, [x/len(dataloader) for x in iter_losses], avg_dice


def _set_criterion(name, params=None):
    """Set the loss function based on name.


    Returns: The correct loss function.

    """
    print("Using loss function: {}".format(name))
    if name == "CombinedDLCrossent":
        return CombinedDLCrossent()
    elif name == "DiceLoss":
        return DiceLoss()
    elif name == "Crossent":
        return torch.nn.BCELoss()
    elif name == "Alternating":
        return [torch.nn.BCEWithLogitsLoss(), DiceLoss()]
    elif name == "IoULoss":
        return IoULoss()
    elif name == "CombinedIoUCrossent":
        return CombinedIoUCrossent()
    elif name == "CombinedWDLCrossent":
        return CombinedWDLCrossent(dice_weight=params['dice_weight'],
                                   crossent_weight=params['crossent_weight'])


def train_model(args):

    if args.hypertune:
        hpt = hypertune.HyperTune()

        if hpt.trial_id is not None:
            save_path = os.path.join(args.checkpoint_path, str(hpt.trial_id))
        else:
            save_path = args.checkpoint_path
    else:
        save_path = args.checkpoint_path

    if not _check_checkpoint_dir(save_path):
        return

    torch.manual_seed(args.seed)

    # Get the dataset
    if args.gcloud:
        train_path = data.download_from_gloud(args.train_dataset_name,
                                 args.storage_bucket_id)
        test_path = data.download_from_gloud(args.test_dataset_name,
                                             args.stotage_bucket_id)

    else:
        train_path = args.train_path
        test_path = args.test_path

    trainloader = data.load_patch_set(train_path, args.seed, args.batch_size,
                                      training=True)

    testloader = data.load_patch_set(test_path, args.seed, args.batch_size,
                                     training=True)

    # Create the model

    model = ITErRoot(args.iterations)

    criterion_params = None

    if args.criterion == "CombinedWDLCrossent":
        criterion_params = {}
        if args.dice_weight:
            criterion_params['dice_weight'] = args.dice_weight
        else:
            criterion_params['dice_weight'] = 1.0

        if args.crossent_weight:
            criterion_params['crossent_weight'] = args.crossent_weight
        else:
            criterion_params['crossent_weight'] = 1.0
        print(criterion_params)

    criterion = _set_criterion(args.criterion, params=criterion_params)

    model, criterion = _set_devices(model, criterion)

    optimizers = []
    if isinstance(model, torch.nn.DataParallel):
        optimizers.append(torch.optim.Adam(model.module.main_net.parameters(),
                                           lr=args.lr))

        for net in model.module.secondary_nets:
            optimizers.append(torch.optim.Adam(net.parameters(),
                                               lr=args.lr))

    else:
        optimizers.append(torch.optim.Adam(model.main_net.parameters(),
                                           lr=args.lr))

        for net in model.secondary_nets:
            optimizers.append(torch.optim.Adam(net.parameters(),
                                               lr=args.lr))

    schedulers = []
    for opt in optimizers:
        lmbda = lambda epoch: args.lr_decay
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt,
                                                              lr_lambda=lmbda)
        schedulers.append(scheduler)

    early_stopper = EarlyStopperSingle(args.stopping_epochs,
                                       args.stopping_patience,
                                       args.stopping_tolerance)

    best_dice = 0.0

    for epoch in range(args.epochs):

        train_loss, train_losses = _train_step(trainloader, model,
                                               optimizers, epoch,
                                               args.epochs, criterion,
                                               num_minibatch=args.num_minibatch)

        eval_loss, eval_losses, dice_score = _eval_step(testloader, model,
                                                        criterion, epoch)

        for scheduler in schedulers:
            scheduler.step()

        print("Train Loss: {}  Eval Loss: {}".format(train_loss, eval_loss))
        print("Eval Dice: {}".format(dice_score))
        print("\n")

        is_best = False
        if dice_score > best_dice:
            best_dice = dice_score
            is_best = True

        if args.job_dir:
            job_dir = args.job_dir
        else:
            job_dir = None

        _save_checkpoint(save_path,
                         model,
                         optimizers,
                         train_losses,
                         eval_losses,
                         epoch,
                         dice_score,
                         is_best,
                         job_dir)

        # Hyperparmeter tuning
        if args.hypertune:
            hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag='DiceScore',
                    metric_value=dice_score,
                    global_step=epoch
                )
        if early_stopper.update(eval_loss, epoch):
            print("Early stopping triggered at epoch {}".format(epoch))
            return


def _save_checkpoint(path, model, optimizer, train_loss, eval_loss, epoch,
                     dice_score, is_best, job_dir=None):
    """Save a model checkpoint.

    Args:
        path (string): The path to store checkpoint data in.
        model (model): The model to save.
        optimizer (torch.optim): The optimizer.
        train_loss (float): The current training loss.
        eval_loss (float): The current eval loss.
        epoch (int): The current epoch.
        dice_score (float): The dice score at this epoch.
        best_loss (bool): If this is the best eval loss so far.
        job_dir (string): The job dir for google cloud saving. Default is None,
                          which will ignore the cloud save bit.

    Returns: None

    """
    checkpoint_path = os.path.join(path, "model_checkpoint.pt")
    best_path = os.path.join(path, "best_model.pt")

    loss_log_path = os.path.join(path, "train_loss.csv")
    eval_log_path = os.path.join(path, "eval_loss.csv")

    optimizer_states = [opt.state_dict() for opt in optimizer]

    if isinstance(model, torch.nn.DataParallel):
        iters = model.module.iters
    else:
        iters = model.iters

    torch.save({
            "epoch": epoch,
            "iterations": iters,
            "loss": train_loss,
            "state": model.state_dict(),
            "optimizer_state": optimizer_states
            }, checkpoint_path)

    if is_best:
        torch.save({
            "epoch": epoch,
            "iterations": iters,
            "loss": train_loss,
            "state": model.state_dict(),
            "optimizer_state": optimizer_states
            }, best_path)

        _save_to_cloud(job_dir, best_path)

    with open(loss_log_path, 'a') as csvfile:
        writer = csv.writer(csvfile)

        row = [epoch] + train_loss + [dice_score]
        writer.writerow(row)

    with open(eval_log_path, 'a') as csvfile:
        writer = csv.writer(csvfile)

        row = [epoch] + eval_loss + [dice_score]
        writer.writerow(row)

    _save_to_cloud(job_dir, checkpoint_path)
    _save_to_cloud(job_dir, loss_log_path)
    _save_to_cloud(job_dir, eval_log_path)


def _save_to_cloud(job_dir, path):
    """Save the data to the cloud.

    Args:
        job_dir (str): The job dir from the job submission.
        path (str): The path to the file to save.

    Returns: {% TODO %}

    """
    if job_dir is None:
        return

    print("Saving {} to {}".format(path, job_dir))


    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]

    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob('{}/{}'.format(bucket_path, path))
    print("bucket_path")
    blob.upload_from_filename(path)
    print("Done upload of {}".format(path))


def _check_checkpoint_dir(path):
    """Make sure the checkpointing directory is all set.

    Args:
        path (string): The path to the checkpoint dir.

    Returns: False if there is a problem.

    """

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(path):
        if os.listdir(path):
            resp = input("{} contains saved models, overwrite?"
                         "(y/n)".format(path))

            if resp.lower() != 'y':
                return False
            else:
                try:

                    check_path = os.path.join(path,
                                              "model_checkpoint.pt")
                    best_path = os.path.join(path,
                                             "best_model.pt")
                    test_path = os.path.join(path,
                                             "eval_loss.csv")

                    train_path = os.path.join(path,
                                              "train_loss.csv")

                    if os.path.exists(check_path):
                        os.remove(check_path)
                    if os.path.exists(best_path):
                        os.remove(best_path)

                    if os.path.exists(test_path):
                        os.remove(test_path)

                    if os.path.exists(train_path):
                        os.remove(train_path)

                except Exception as e:
                    print("Error removing files.")
                    return False

        return True


def get_args():
    """Get commandline arguments with argparse.
    Returns: The args dictionary

    """
    parser = argparse.ArgumentParser(description='Train the network.\n\n'
                                                 'This can be done with Google'
                                                 'Cloud Services using the '
                                                 'corresponding flags.  If '
                                                 'the --gcloud flag is not '
                                                 'set, you must specify the '
                                                 '--dataset-path flag.')

    # Should be handled by the AI platform
    parser.add_argument('--job-dir', help="GCS location to write checkpoints"
                        "and export models.")

    parser.add_argument('--model-name', type=str,
                        default="ITErRootModel",
                        help="What to name the saved model file.")

    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help="Input batch size for training (default: 4)")

    parser.add_argument('--num-minibatch',
                        type=int,
                        default=1,
                        help="The number of minibatches, multiply this by the"
                        "batch size to get the full simulated batch size."
                        "(default 1)")

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs for training (default: 10).'
                        )

    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate (default: 0.01)'
                        )

    parser.add_argument('--lr-decay',
                        type=float,
                        default=0.95,
                        help='Multiplicative learning rate decay (default: 0.95)'
                        )

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed (default: 42)'
                        )

    parser.add_argument('--iterations',
                        type=int,
                        default=3,
                        help="The number of iterations in the network "
                        "(default: 3)")

    parser.add_argument('--criterion',
                        type=str,
                        default="DiceLoss",
                        help="The loss function to use (default: DiceLoss)")


    parser.add_argument('--stopping-patience',
                        type=str,
                        default=10,
                        help="The number of epochs to wait before enabling"
                        "early stopping. (default: 10)")

    parser.add_argument('--stopping-tolerance',
                        type=float,
                        default=0.0,
                        help="The amount of difference in the new and old"
                        "evaluation metric to consider stopping. "
                        "(default: 0.0)")

    parser.add_argument('--stopping-epochs',
                        type=int,
                        default=10,
                        help="The number of epochs to use for the average."
                             "(default: 10)")

    parser.add_argument('--dice-weight',
                        type=float,
                        help="The weighting for dice score if using weighted"
                             "loss")

    parser.add_argument('--crossent-weight',
                        type=float,
                        help="The weighting for the crossent score if using"
                             "weighted loss")

    parser.add_argument("--hypertune",
                        action="store_true",
                        help="Add this flag to use Google Cloud Hyperparameter"
                             "Tuning")

    parser.add_argument("--gcloud",
                        action="store_true",
                        help="Add this flag to use Google Cloud Services.")

    parser.add_argument("--train-path",
                        type=str,
                        help="The path to the dataset for training.")

    parser.add_argument("--test-path",
                        type=str,
                        help="The path to the dataset for testing.")

    parser.add_argument("--checkpoint-path",
                        type=str,
                        default="./model_checkpoints/test_model",
                        help="The path to store model checkpoints.")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    train_model(args)


if __name__ == "__main__":
    main()
