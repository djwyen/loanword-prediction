import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing.transcriber import Transcriber
from data_processing.bccwj_dataset import BCCWJDataset, split_pared_bccwj
from data_processing.word_loaner import WordLoaner
from models.model import Encoder, Decoder, AutoEncoder

SEED = 888

BATCH_SIZE = 64
NUM_PHONETIC_FEATURES = 22
HIDDEN_DIM = 32
MAX_SEQ_LEN_NO_STOP = 20
MAX_SEQ_LEN_WITH_STOP = MAX_SEQ_LEN_NO_STOP + 1 # due to the added stop token at the end

class TestWordLoaning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.t = Transcriber()
        # self.wl = WordLoaner()

    def test_greedy_select_segment(self):
        # a word's exact representation should be decoded as the word itself, as it should have distance 0
        # at each segment from the correct one. Just a sanity check.
        fv = self.t.ipa_to_feature_vectors('oɾoɕiɯɾi')
        transcription = ''
        for seg_fv in fv:
            transcription += self.t.greedy_select_segment(seg_fv, weighted=True)
        self.assertEqual(transcription, 'oɾoɕiɯɾi')


if __name__ == '__main__':
    unittest.main()