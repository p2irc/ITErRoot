"""
File Name: plot_loss.py

Authors: Kyle Seidenthal

Date: 19-11-2020

Description: A script to plot the training and testing loss over time.

"""

import matplotlib.pyplot as plt
import csv
import numpy as np

import argparse
import os

def read_loss_file(path):
    """Read the given loss csv file and process its data into lists that can be
    plotted by matplotlib.

    Args:
        path (string): The path to the file to be read.

    Returns: A list of lists, one list for each subnetwork containing the loss
             values over time.

    """
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        data = []

        for row in reader:
            # Ignore the epoch numbers
            if len(data) == 0:
                data = [[] for _ in row[1:]]

            for i in range(1, len(row)):
                data[i-1].append(float(row[i]))

        return data


def plot_loss(train_losses, test_losses, out_file, best_epoch=None):
    """Plot the losses and save.

    Args:
        train_losses (list): The training losses to plot.
        test_losses (list): The testing losses to plot.

    Returns: {% TODO %}

    """

    num_plots = len(train_losses)

    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 15))

    i = 0
    for data in train_losses:

        if best_epoch is not None:
            axs[i].axvline(best_epoch, label="best", color='red')

        if i == len(train_losses) - 1:
            i += 1
            continue

        label = "train_" + str(i)
        axs[i].plot(data, label=label)

        i += 1

    i = 0
    for data in test_losses:
        label = "test_" + str(i)
        axs[i].plot(data, label=label)

        axs[i].legend()
        axs[i].set_ylim([0, 0.7])

        if i == len(test_losses) - 1:
            axs[i].set_title("Dice Score")
            axs[i].set_ylim([0.8, 1.0])
        else:
            axs[i].set_title("Loss for Net " + str(i))

        i += 1


    # Set common labels
    fig.text(0.5, 0.04, 'Epoch', ha='center', va='center')
    fig.text(0.06, 0.5, 'Loss', ha='center', va='center',
             rotation='vertical')
    fig.suptitle("Training and Testing Loss")
    plt.savefig(out_file)

    fig, axs = plt.subplots(1, 1, figsize=(15, 15))

    axs.set_xlabel("loss")
    axs.set_ylabel("epoch")

    fig.suptitle("Overall Model Loss")
    axs.plot(train_losses[-2], label="train")
    axs.plot(test_losses[-2], label="test")

    if best_epoch is not None:
            axs.axvline(best_epoch, label="best", color='red')

    axs.legend()


    large_name = os.path.splitext(out_file)[0] + "_large.png"


    plt.savefig(large_name)

def main():
    parser = argparse.ArgumentParser(description="Plot Training Loss")

    parser.add_argument('train_file', help="The path to the csv file "
                        "contining training losses")

    parser.add_argument('test_file', help="The path to the csv file "
                        "containing test losses.")

    parser.add_argument("out_file", help="The path to the output file.")

    parser.add_argument("--best-epoch",
                        type=int,
                        help="The epoch where the best model was saved.")

    args = parser.parse_args()

    if not args.train_file or not args.test_file:
        parser.print_help()
        return

    train_data = read_loss_file(args.train_file)
    test_data = read_loss_file(args.test_file)

    if args.best_epoch is not None:
        best_epoch = args.best_epoch

    plot_loss(train_data, test_data, args.out_file, best_epoch=best_epoch)


if __name__ == "__main__":
    main()
