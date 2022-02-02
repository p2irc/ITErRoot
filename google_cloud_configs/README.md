# Google Cloud Configs
These are configuration files for running jobs on Google Cloud Platform.  Their general format is outlined here:(https://cloud.google.com/ai-platform/training/docs/training-jobs)[https://cloud.google.com/ai-platform/training/docs/training-jobs].

# `expanded_patch_training_config.yaml`
This configures a simple training job for a sinlge model.  It creates a custom run config using 4 NVIDIA-Tesla T4 GPUs.

# `hyperparam_config.yaml`
This configures a hyperparameter tuning job which will run up to 30 training trials, maximizing on Dice Score.  It outlines a number of details for the hyperparameter search.  It also uses 4 NVIDIA-Tesla T4 GPUs.  More information about hyperparameter tuning can be found at (https://cloud.google.com/ai-platform/training/docs/using-hyperparameter-tuning)[https://cloud.google.com/ai-platform/training/docs/using-hyperparameter-tuning].


