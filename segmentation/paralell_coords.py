"""
File Name: paralell_coords.py

Authors: Kyle Seidenthal

Date: 24-02-2021

Description: Make a parallel coords plot for the hyperparameters.

"""

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import hiplot as hip
import plotly.express as px



def read_json(path):
    with open(path) as jsonfile:
        data = json.load(jsonfile)

    return data

def get_args():

   parser = argparse.ArgumentParser(description="Create a parallel coords plot from "
                                     "hyperperam data.")

   parser.add_argument("hyperparam_json",
                       type=str,
                       default=None,
                       help="The path to the hyperparameter json.")

   args = parser.parse_args()

   return args


def get_dataframe(json_data):

    trials = json_data['trials']

    df = pd.DataFrame(columns=["trial-id", "crossent-w", "lr",
                               "lr-decay", "stop-ep",
                               "stop-pat", "stop-tol",
                               "dice", "training_step"])

    for trial in trials:
        trial_id = trial['trialId']

        crossent_weight = float(trial['hyperparameters']['crossent-weight'])
        lr = float(trial['hyperparameters']['lr'])
        lr_decay = float(trial['hyperparameters']['lr-decay'])
        stopping_epochs = float(trial['hyperparameters']['stopping-epochs'])
        stopping_patience = float(trial['hyperparameters']['stopping-patience'])
        stopping_tolerance = float(trial['hyperparameters']['stopping-tolerance'])

        dice_score = trial["finalMetric"]["objectiveValue"]
        training_step = trial["finalMetric"]['trainingStep']


        df = df.append({
                'trial-id': trial_id,
                'crossent-w': crossent_weight,
                'lr': lr,
                'lr-decay': lr_decay,
                'stop-ep': stopping_epochs,
                'stop-pat': stopping_patience,
                'stop-tol': stopping_tolerance,
                'dice': dice_score,
                'training-step': training_step
                }, ignore_index=True)

    return df



def main():
    args = get_args()

    data = read_json(args.hyperparam_json)

    df = get_dataframe(data)
    df.to_csv("/out/path/figures/hyperparam-trials.csv")
    print(df)
    fig = px.parallel_coordinates(df,color=df['dice'].astype('category').cat.codes)
    fig.show()
    hyper_plot = hip.Experiment.from_dataframe(df)
    hyper_plot.to_html("hyperparam_plot.html")

if __name__ == "__main__":
    main()
