# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:49:03 2024

@author: an_fab
"""

import numpy as np
from tensorflow.keras.utils import Sequence

class RandomBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(X))

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_indices = np.random.choice(self.indices, self.batch_size, replace=False)
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        # Shuffle indices after each epoch
        np.random.shuffle(self.indices)