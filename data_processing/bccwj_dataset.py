'''
Custom Dataset/DataLoader for working with the (pared) BCCWJ vocabulary.

Following this: (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
'''

import os
import csv
from math import floor
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

# path to the pared BCCWJ dataset that is created by `process_bccwj.py`
PATH_TO_PROCESSED_CSV = "data/BCCWJ/pared_BCCWJ.csv"
PATH_TO_PROCESSED_GZ = "data/BCCWJ/fv_pared_BCCWJ.gz"

NUM_PHONETIC_FEATURES = 22 # Panphon by default gives you 24 features, but the last two corresond to tonal features so I drop them
CATEGORIES_PER_FEATURE = 3 # the categories being {+, -, 0} in that order
END_MULTIHOT_FV = [0] * NUM_PHONETIC_FEATURES * CATEGORIES_PER_FEATURE
PAD_MULTIHOT_FV = [1] * NUM_PHONETIC_FEATURES * CATEGORIES_PER_FEATURE

# TODO could refactor to transcribe from kana to ipa in the Dataset, which would allow passing transcription broadness as a flag to the constructor for the Dataset. Pared_BCCWJ would just be to pick out the relevant words, and not to pretranscribe them.

class BCCWJDataset(Dataset):
    def __init__(self, indices=list(range(36396)), max_seq_len=20): # hardcoded values true as of May 11
        # note: The max_seq_len here is the length of the longest sequence without the end-of-word token,
        #       which is something this module adds itself.
        #       However, the property max_seq_len it will be constructed with will be the correct length
        #       of the longest sequence, which includes the +1.
        self.vocab_nparray = np.loadtxt(PATH_TO_PROCESSED_GZ)
        # by the way that the np array was saved, each entry is a flattened version
        # of the word; ie it was flattened from (MAX_LEN, N_FEATURES) to just (MAX_LEN * N_FEATURES)
        # so you need to reshape to recover it, which we can perform in __item__
        self.indices = indices # the set of indices this Dataset covers
        self.max_seq_len = max_seq_len + 1 # to accommodate the appended end-of-word tokens

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        '''
        Returns items as NumPy Arrays, padded to be of the max seq len
        '''
        # TODO assert index is less than length?
        true_idx = self.indices[idx] # the index of the item in pared_BCCWJ we are retrieving
        # we have to unflatten the retrieved word from a single vector into a list of segmental feature vectors
        flat_word = self.vocab_nparray[true_idx, :]
        # unflatten to be a list of segment multihot feature vectors.
        feature_vectors = flat_word.reshape((self.max_seq_len, CATEGORIES_PER_FEATURE*NUM_PHONETIC_FEATURES))

        return feature_vectors


def split_pared_bccwj(seed, frac, max_seq_len=None):
    """
    Splits the pared bccwj data into a test set and a training set such that `f` in [0, 1]
    of the data becomes test and the rest is kept as training.
    `seed` is the random seed to use for reproducibility.
    `max_seq_len` is the maximum sequence length, excluding the end-of-word token that will be appended
    """
    n_words = np.loadtxt(PATH_TO_PROCESSED_GZ).shape[0]
    n_train = floor(frac * n_words)
    n_test = n_words - n_train
    split = [n_train, n_test]

    train_indices, test_indices = random_split(range(n_words), split,
                                               generator=torch.Generator().manual_seed(seed))

    return BCCWJDataset(train_indices, max_seq_len), BCCWJDataset(test_indices, max_seq_len)
