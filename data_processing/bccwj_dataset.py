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
END_FV = [2] * NUM_OF_PANPHON_FEATURES

# TODO could refactor to transcribe from kana to ipa in the Dataset, which would allow passing transcription broadness as a flag to the constructor for the Dataset. Pared_BCCWJ would just be to pick out the relevant words, and not to pretranscribe them.

def length_of_ipa(ipa):
    '''
    Quick helper function to compute the length of an ipa string by removing extraneous segments
    '''
    return len(ipa) - ipa.count('ː') - ipa.count('ʲ') - ipa.count('ç') - (2*ipa.count('d͡ʑ')) - (2*ipa.count('d͡z')) - (2*ipa.count('t͡ɕ')) - (2*ipa.count('t͡s')) - ipa.count('ɰ̃') - ipa.count('ĩ')


class BCCWJDataset(Dataset):
    def __init__(self, indices=None, max_seq_len=None):
        # note: The max_seq_len here is the length of the longest sequence without the end-of-word token,
        #       which is something this module adds itself.
        #       However, the property max_seq_len it will be constructed with will be the correct length
        #       of the longest sequence, which includes the +1.
        self.vocab_df = pd.read_csv(PATH_TO_PROCESSED_CSV)
        self._t = Transcriber()
        if indices is not None:
            self.indices = indices # the set of indices this Dataset covers
        else:
            self.indices = range(len(self.vocab_df))
        
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len + 1 # to accommodate the appended end-of-word tokens
        else:
            # find the longest seq in this dataset ourselves
            self.max_seq_len = 0
            for i, row in self.vocab_df.iterrows():
                ipa = row['ipa']
                length = length_of_ipa(ipa)
                if length > self.max_seq_len:
                    self.max_seq_len = length
            self.max_seq_len += 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        '''
        Returns items as NumPy Arrays, padded to be of the max seq len
        TODO look at having an end of sequence token before padding?
        '''
        # assert index is less than length?
        true_idx = self.indices[idx] # the index of the item in pared_BCCWJ we are retrieving

        ipa = self.vocab_df.at[true_idx, 'ipa']

        # use Transcriber/panphon to create the more informative info about this word
        # segments = self._t.ipa_to_panphon_segments(ipa)
        feature_vectors = self._t.ipa_to_feature_vectors(ipa)
        
        # pad sequence with dead all-zeroes segments up to the max seq len
        length_diff = (self.max_seq_len - 1) - length_of_ipa(ipa)
        # the -1 is to account for the fact that max_seq_len has been increased by 1 for the end-of-word token
        # We must pad by one less than this length difference.
        for _ in range(length_diff):
            feature_vectors.append(PAD_FV)
        feature_vectors.append(END_FV)

        return np.array(feature_vectors)


def split_pared_bccwj(seed, frac, max_seq_len=None):
    """
    Splits the pared bccwj data into a test set and a training set such that `f` in [0, 1]
    of the data becomes test and the rest is kept as training.
    `seed` is the random seed to use for reproducibility.
    `max_seq_len` is the maximum sequence length, excluding the end-of-word token that will be appended
    """
    n_words = len(BCCWJDataset())
    n_train = floor(frac * n_words)
    n_test = n_words - n_train
    split = [n_train, n_test]

    train_indices, test_indices = random_split(range(n_words), split,
                                               generator=torch.Generator().manual_seed(seed))

    return BCCWJDataset(train_indices, max_seq_len), BCCWJDataset(test_indices, max_seq_len)
