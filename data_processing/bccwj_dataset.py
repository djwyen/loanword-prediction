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
from torch.utils.data import Dataset, DataLoader, random_split
from .transcriber import Transcriber
from .word import Word

# path to the pared BCCWJ dataset that is created by `process_bccwj.py`
PATH_TO_PROCESSED_CSV = "data/BCCWJ/pared_BCCWJ.csv"

NUM_OF_PANPHON_FEATURES = 24 # there are 24 features output by panphon by default transcription
PAD_FV = [0] * NUM_OF_PANPHON_FEATURES

# TODO could refactor to transcribe from kana to ipa in the Dataset, which would allow passing transcription broadness as a flag to the constructor for the Dataset. Pared_BCCWJ would just be to pick out the relevant words, and not to pretranscribe them.

def length_of_ipa(ipa):
    '''
    Quick helper function to compute the length of an ipa string by removing extraneous segments
    '''
    return len(ipa) - ipa.count('ː') - ipa.count('ʲ') - ipa.count('ç') - (2*ipa.count('d͡ʑ')) - (2*ipa.count('t͡ɕ')) - (2*ipa.count('t͡s')) - ipa.count('ɰ̃') - ipa.count('ĩ')


class BCCWJDataset(Dataset):
    def __init__(self, indices=None, max_seq_len=None):
        self.vocab_df = pd.read_csv(PATH_TO_PROCESSED_CSV)
        self._t = Transcriber()
        if indices is not None:
            self.indices = indices # the set of indices this Dataset covers
        else:
            self.indices = range(len(self.vocab_df))
        
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = 0
            # find the longest seq in this dataset
            for i, row in self.vocab_df.iterrows():
                ipa = row['ipa']
                length = length_of_ipa(ipa)
                if length > self.max_seq_len:
                    self.max_seq_len = length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        '''
        Returns items as NumPy Arrays, padded to be of the max seq len
        TODO look at having an end of sequence token before padding?
        '''
        # assert index is less than length?
        true_idx = self.indices[idx] # the index of the item in pared_BCCWJ we are retrieving

        word = self.vocab_df.at[true_idx, 'word']
        kana = self.vocab_df.at[true_idx, 'kana']
        ipa = self.vocab_df.at[true_idx, 'ipa']
        origin = self.vocab_df.at[true_idx, 'origin']

        # use Transcriber/panphon to create the more informative info about this word
        # segments = self._t.ipa_to_panphon_segments(ipa)
        feature_vectors = self._t.ipa_to_feature_vectors(ipa)

        # pad sequence with dead all-zeroes segments up to the max seq len
        length_diff = self.max_seq_len - length_of_ipa(ipa)
        feature_vectors += PAD_FV * length_diff

        return np.array(feature_vectors), Word(true_idx, word, kana, origin, ipa)


def split_pared_bccwj(seed, frac):
    """
    Splits the pared bccwj data into a test set and a training set such that `f` in [0, 1]
    of the data becomes test and the rest is kept as training.
    `seed` is the random seed to use for reproducibility.
    """
    n_words = len(BCCWJDataset())
    n_train = floor(frac * n_words)
    n_test = n_words - n_train
    split = [n_train, n_test]

    assert(sum(split) == n_words, "Split does not use all words")

    train_indices, test_indices = random_split(range(n_words), split,
                                               generator=torch.Generator().manual_seed(seed))

    return BCCWJDataset(train_indices), BCCWJDataset(test_indices)
