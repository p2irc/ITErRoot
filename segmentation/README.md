# Segmentation
This module contains everything needed to run the segmentation network.

## `data_utils`
This module contains various scripts used to manage the datasets for training and evaluating the network.

## `dataset_csvs`
The csv files that record the images used for the patch dataset.

## `network_modules`
This module contains the main definition of the network and all of its pieces.

## `compute_data_mean.py`
This script can be used to compute the data mean of your dataset for use with Normalization.

```
usage: compute_data_mean.py [-h] dataset_path

Find the mean and std of the dataset.

positional arguments:
  dataset_path  Path to the directory containing the images/ and masks/ directories.

optional arguments:
  -h, --help    show this help message and exit
```

## `Docker_images`
This folder contains docker image setups for using the model with Google cloud or Docker

## `per_species_results.py`
This script breaks down the results of a dataset by species.

## `predict.py`
This script is used to create predicted segmentations of a dataset.

```
usage: predict.py [-h] [--seed SEED] [--batch-size BATCH_SIZE] [--checkpoint-path CHECKPOINT_PATH] [--gcloud-checkpoint-path GCLOUD_CHECKPOINT_PATH] [--dataset_path DATASET_PATH] [--gcloud] [--patch_set]
                  [--gcloud_set_name GCLOUD_SET_NAME] [--storage_bucket_id STORAGE_BUCKET_ID]
                  eval_out

Evaluate a model checkpoint.This can be done with GoogleCloud Services using the corresponding flags. If the --gcloud flag is not set, you must specify the --dataset-path flag.

positional arguments:
  eval_out              Where to put output images

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           The random pytorch seed.
  --batch-size BATCH_SIZE
                        The batch size to use.
  --checkpoint-path CHECKPOINT_PATH
                        The path to the model checkpoint.
  --gcloud-checkpoint-path GCLOUD_CHECKPOINT_PATH
                        The path to the checkpoint in GCloud.
  --dataset_path DATASET_PATH
                        The path to the dataset to use if not usingGcloud.
  --gcloud              Add this flag to use gcloud.
  --patch_set           Add this flag to predict for a set of patches rather than full images.
  --gcloud_set_name GCLOUD_SET_NAME
                        The name of the dataset file to download from gcloud.
  --storage_bucket_id STORAGE_BUCKET_ID
                        The storage bucket id for the dataset to download from gcloud.
```

## `train.py`
This script is used to train the network.

```
usage: train.py [-h] [--job-dir JOB_DIR] [--model-name MODEL_NAME] [--batch-size BATCH_SIZE] [--num-minibatch NUM_MINIBATCH] [--epochs EPOCHS] [--lr LR] [--lr-decay LR_DECAY] [--seed SEED]
                [--iterations ITERATIONS] [--criterion CRITERION] [--stopping-patience STOPPING_PATIENCE] [--stopping-tolerance STOPPING_TOLERANCE] [--stopping-epochs STOPPING_EPOCHS]
                [--dice-weight DICE_WEIGHT] [--crossent-weight CROSSENT_WEIGHT] [--hypertune] [--gcloud] [--train-path TRAIN_PATH] [--test-path TEST_PATH] [--checkpoint-path CHECKPOINT_PATH]

Train the network. This can be done with GoogleCloud Services using the corresponding flags. If the --gcloud flag is not set, you must specify the --dataset-path flag.

optional arguments:
  -h, --help            show this help message and exit
  --job-dir JOB_DIR     GCS location to write checkpointsand export models.
  --model-name MODEL_NAME
                        What to name the saved model file.
  --batch-size BATCH_SIZE
                        Input batch size for training (default: 4)
  --num-minibatch NUM_MINIBATCH
                        The number of minibatches, multiply this by thebatch size to get the full simulated batch size.(default 1)
  --epochs EPOCHS       Number of epochs for training (default: 10).
  --lr LR               Learning rate (default: 0.01)
  --lr-decay LR_DECAY   Multiplicative learning rate decay (default: 0.95)
  --seed SEED           Random seed (default: 42)
  --iterations ITERATIONS
                        The number of iterations in the network (default: 3)
  --criterion CRITERION
                        The loss function to use (default: DiceLoss)
  --stopping-patience STOPPING_PATIENCE
                        The number of epochs to wait before enablingearly stopping. (default: 10)
  --stopping-tolerance STOPPING_TOLERANCE
                        The amount of difference in the new and oldevaluation metric to consider stopping. (default: 0.0)
  --stopping-epochs STOPPING_EPOCHS
                        The number of epochs to use for the average.(default: 10)
  --dice-weight DICE_WEIGHT
                        The weighting for dice score if using weightedloss
  --crossent-weight CROSSENT_WEIGHT
                        The weighting for the crossent score if usingweighted loss
  --hypertune           Add this flag to use Google Cloud HyperparameterTuning
  --gcloud              Add this flag to use Google Cloud Services.
  --train-path TRAIN_PATH
                        The path to the dataset for training.
  --test-path TEST_PATH
                        The path to the dataset for testing.
  --checkpoint-path CHECKPOINT_PATH
                        The path to store model checkpoints.
```

## `paralell_coords.py`
This is a simple script which creates a parallel coordinates plot from a set of data.

## `summarize.py` 
This script includes code which should print out a text summary of the model structure.

## `compute_metrics.py`
This script will compute the evaluation metrics for a set of predicted and ground truth images, and dump them to a JSON file in the specified output directory.

```
usage: Get evaluation metrics for images. [-h] pred_path gt_path out_path

positional arguments:
  pred_path   The path to the predicted segmentations.
  gt_path     The path to the ground truth segmentations.
  out_path    The path to store the metrics json file.

optional arguments:
  -h, --help  show this help message and exit
```
