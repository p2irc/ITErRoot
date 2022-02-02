# Welcome to ITErRoot

ITErRoot is a deep neural network architecture, which performs Iterative Topological Extraction of Roots from 2D images.

# Quick Start
The Fastest way to get training and predicting with ITErRoot

## Clone the repository
```
git clone https://github.com/p2irc/ITErRoot.git
```

## Create and activate the conda environment
This will require you to have `conda` installed (https://www.anaconda.com/)

```
conda env create -f environment.yml
conda activate iter-root
```

## To Train ITErRoot On An Image Dataset
Use the `segmentation/train.py` script.
Note that the path to the training and testing sets must contain subdirectories `imgs/` for the input images and `masks/` for the ground truth masks.
```
cd segmentation
python train.py --model-name "My-Model" --train-path "/path/to/train/imgs" --test-path "path/to/test/imgs" --checkpoint-path "/path/to/store/model/checkpoints/"
```

For more complex setup, see `python train.py -h`

The trained model should appear in the checkpoint path as a `.pt` file.

## To Evaluate A Trained Model
First we predict the segmentations using the `predict.py` script:
```
python predict.py /path/to/store/output/segmentations/ --checkpoint-path /path/to/model/checkpoint/ --dataset_path "/path/to/input/images/"
```

For more complex setup, see `python predict.py -h`

The predicted segmentations should appear in the `/path/to/store/output/segmentations/`.

## See how it performed
We can compute a number of evaluation metrics using the `compute_metrics.py` script.

```
python compute_metrics.py /path/to/predictions /path/to/gt /path/to/store/results
```

This will give us values for each of the evaluation metrics in the `/path/to/store/results`.

# Contents

## google_cloud_configs
Some things which configure google cloud jobs for training.  Probably won't need to use these if you don't use Google Cloud Platform, but they are here just in case.

## Segmentation
This is it, the event of the evening.  This folder contains the model definitions and scripts used for ITErRoot.  More details within.

## Docker
The pipeline and each of its components can use Docker to keep a consistent environment.  This was done for use with Google cloud platform, but is not necessary for running locally (though it would still work).

Modules contain their own Dockerfiles, which can be used for various training and evaluation tasks.


To build an image, from the root repository directory:
`sudo docker build -f path/to/Dockerfile.name -t $IMAGE_NAME .`

To run a build image:
`sudo docker run --runtime=nvidia $IMAGE_NAME [program-args]`
