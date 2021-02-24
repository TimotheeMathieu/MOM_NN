"""Robust Neural Network meta estimator."""

# Author: Timothee Mathieu
# License: BSD 3 clause

import numpy as np
import warnings
from scipy.stats import iqr

from sklearn.base import BaseEstimator, clone
from sklearn.utils import (
    check_random_state,
    check_array,
    check_consistent_length,
    shuffle,
)
from tensorflow.keras import backend as Ke
import time

# TODO : optimize computation of the loss.

class MOM():
    def __init__(self, K=3):
        self.K = K
        self.blocks = None
    def make_blocks(self, X):
        x = X.flatten()
        # Sample a permutation to shuffle the data.
        perm = np.random.permutation(len(x))
        self.blocks = np.array_split(perm, self.K)
    def estimate(self, X):
        if self.blocks is None:
            self.make_blocks(X)
        x = X.flatten()

        # Compute the mean of each block
        means_blocks = [np.mean([x[f] for f in ind]) for ind in self.blocks]

        # Find the indice for which the mean of block is the median-of-means.
        indice = np.argsort(means_blocks)[int(np.floor(len(means_blocks) / 2))]
        return means_blocks[indice], indice




class MOM_model(BaseEstimator):
    """Meta algorithm for robust NN
    Parameters
    ----------

    model : keras model

    loss : a keras loss function

    K : int, default = 3
        number of blocks used in MOM

    max_iter : int, default = 100
        maximum number of iterations

    burn_in: int, default = 0
        number of steps on which we don't update the epoch (and the step size).

    task: string, default='classification'
        task must be either 'classification', 'regression', or 'autoencoder'

    batch_size: int, default = 256
        how much data is fed in one step to the algorithm.

    evaluation_scoring: scoring function, default=None

    random_state: random_state

    verbose: int, default = 0
        verbosity of the algorithm.

    n_steps: int, default = 1
        how many steps to do for each epoch (and each MOM).
    Return
    ------

    """

    def __init__(
        self,
        model,
        loss,
        K=3,
        max_iter=100,
        burn_in=0,
        task="classification",
        batch_size=256,
        evaluation_scoring=None,
        random_state=None,
        verbose=0,
        n_steps=1
    ):
        self.model = model
        self.burn_in = burn_in
        self.loss = loss
        self.K = K
        self.task = task
        self.max_iter = max_iter
        self.batch_size=batch_size
        self.scoring=evaluation_scoring
        self.random_state = random_state
        self.verbose=verbose
        self.n_steps=n_steps

    def fit(self, X, y=None, callbacks=None, val_split=None):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns an estimator trained with RobustWeightedEstimator.
        """
        loss=self.loss
        X = check_array(X)
        if y is not None:
            y = check_array(y, ensure_2d=False)
            check_consistent_length(X, y)
        if self.n_steps is None:
            n_steps = self.K
        else:
            n_steps = self.n_steps

        random_state = check_random_state(self.random_state)
        model=self.model
        mean_estimator = MOM(self.K)

        weights = np.zeros(len(X))


        for epoch in np.arange(0, self.max_iter, n_steps):
            # Compute the loss of each sample
            pred=model.predict(X, batch_size=self.batch_size)
            if self.task == "autoencoder":
                losses=loss(X,pred).numpy()
            else:
                losses=loss(y, pred).numpy()

            mloss, indice_med=mean_estimator.estimate(losses)

            indices_med = mean_estimator.blocks[indice_med]


            if epoch < self.burn_in:
                e=0
            else:
                e=epoch

            model.fit(X[indices_med], y[indices_med], epochs=e+n_steps,
                      batch_size=self.batch_size,initial_epoch=e, verbose=self.verbose,
                      callbacks = callbacks, validation_split=val_split)

            # Use the optimization algorithm of self.base_estimator for one
            # epoch using the previously computed weights.

            # Shuffle the data at each step.
            if self.task == "autoencoder":
                X = shuffle(X, random_state=random_state)
            else:
                X, y = shuffle(
                    X, y, random_state=random_state
                )
            weights[indices_med] += 1

        self.weights_ = weights / len(losses)
        self.model_ = model
        self.n_iter_ = self.max_iter * len(X)
        return self

    def predict(self, X):
        """Predict using the estimator trained with RobustWeightedEstimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples, n_outputs)
            The predicted values.
        """
        return self.model_.predict(X)

    def evaluate(self,X,y):
        # Only for classification for now.
        return np.mean(self.predict(X).argmax(axis=1)==y.argmax(axis=1))
