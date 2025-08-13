# -*- coding: utf-8 -*-
"""Customized dataloader.
"""
import random
import numpy as np
from torch.utils.data import DataLoader

class KTDataloader(DataLoader):

    def __init__(self, dataset, batch_size=128, shuffle=True):
        self.dataset = dataset
        self.num_problems = dataset.num_problems
        self.mode = dataset.mode
        self.batch_size = batch_size
        self.shuffle = True

        if shuffle == True:
            random.shuffle(self.dataset)

        if len(dataset) % batch_size == 0:
            self.num_batch = len(dataset) // batch_size
        else:
            self.num_batch = len(dataset) // batch_size + 1
            if self.mode == "train":
                # Concat
                self.dataset += self.dataset[: batch_size -
                                             len(dataset) % batch_size]

    def __iter__(self):
        for batch_idx in range(self.num_batch):
            start_idx = batch_idx * self.batch_size
            batch_user_ids, batch_sessions = self.dataset[start_idx:
                                                          start_idx +
                                                          self.batch_size]
            batch_sessions = list(zip(*batch_sessions))
            yield np.array(batch_user_ids), tuple(np.array(x) for x
                                                  in batch_sessions)