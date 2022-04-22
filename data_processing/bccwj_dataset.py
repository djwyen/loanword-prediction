'''
Custom Dataset/DataLoader for working with the (pared) BCCWJ vocabulary.

Following this: (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
'''

import os
import csv
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from data_processing import transcriber
from .word import Word

# path to the pared BCCWJ dataset that is created by `process_bccwj.py`
PATH_TO_PROCESSED_CSV = "data/BCCWJ/pared_BCCWJ.csv"

class BCCWJDataset(Dataset):
    def __init__(self):
        self.vocab_df = pd.read_csv(PATH_TO_PROCESSED_CSV)
        self.t = transcriber.Transcriber()

    def __len__(self):
        return self.vocab_df.shape[0]

    def __getitem__(self, idx):
        word = self.vocab_df.at(idx, 'word')
        kana = self.vocab_df.at(idx, 'kana')
        ipa = self.vocab_df.at(idx, 'ipa')
        origin = self.vocab_df.at(idx, 'origin')
        segments = self.t.ipa_to_numpy_array(ipa)
        return Word(idx, word, kana, origin, ipa, segments)
