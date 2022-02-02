"""
File Name: early_stopping.py

Authors: Kyle Seidenthal

Date: 30-11-2020

Description: An early stopping management object.

"""
import collections


class EarlyStopperBase():
    """
    An early stopping mechanism based on tracking the average evaluation loss
    over a number of epochs and comparing it to the current epoch loss.
    """

    def __init__(self, stopping_epochs, patience, tolerance):
        """
        Create an early stopping watcher.

        Args:
            stopping_epochs (int): The number of epochs to track the average
                                   over.
            patience (int): The number of epochs to wait before considering
                            stopping early.
            tolerance (float): The amount difference in loss to consider
                               stopping.
        """

        self.prev_stop_eval = 0
        self.prev_eval_losses = collections.deque(maxlen=stopping_epochs)
        self.stopping_epochs = stopping_epochs
        self.patience = patience
        self.tolerance = tolerance

    def update(self, loss, epoch):
        """Update the early stopping history and evaluate.

        Args:
            loss (float): The current loss.
            epoch (int): The current epoch.

        Returns: True if training should stop.

        """
        self.prev_eval_losses.append(loss)

        print("HELLO")
        print(epoch, self.patience)

        if epoch > self.patience:
            return self._check_stopping()

        else:
            return False

    def _check_stopping(self):
        """ Check if we should stop.
        Returns: True of we should stop.

        """
        cur_stop_eval = sum(self.prev_eval_losses) / len(self.prev_eval_losses)
        diff = cur_stop_eval - self.prev_eval_losses[-1]

        print("Cur Stop Eval: {}".format(cur_stop_eval))
        print("Diff: {}".format(diff))

        print("Tolerance: {}".format(self.tolerance))

        if diff < self.tolerance:
            return True

        else:
            return False


class EarlyStopperSingle():

    def __init__(self, stopping_epochs, patience, tolerance):
        """
        Create an early stopper watcher.  This one compares the current epoch
        loss with the best loss so far, and stops when the difference is more
        than the tolerance.

        Args:
            stopping_epochs: The number of 'bad' epochs before stopping.
            patience: The number of epochs to wait before considering stopping.
            tolerance: The amount of difference between the best and current
                       loss to stop.
        """
        self.best = None
        self.stopping_epochs = stopping_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.bad_epochs = 0

    def update(self, loss, epoch):
        """Update the early stopping history and evaluate.

        Args:
            loss (float): The current loss.
            epoch (int): The current epoch.

        Returns: True if training should stop.

        """
        if self.best is None:
            self.best = loss

        if epoch > self.patience:
            if loss > self.best + self.tolerance:
                self.bad_epochs += 1

                if self.bad_epochs == self.stopping_epochs:
                    return True

        if loss < self.best:
            self.best = loss

        return False
