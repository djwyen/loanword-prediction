import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing.bccwj_dataset import BCCWJDataset, split_pared_bccwj
from models.model import Encoder, Decoder, AutoEncoder

SEED = 888
PATH_TO_OUTPUT_CSV = "data/BCCWJ/pared_BCCWJ.csv"
OUTPUT_CSV_LENGTH = 36396

BATCH_SIZE = 64
NUM_PANPHON_FEATURES = 24
HIDDEN_DIM = 32
MAX_SEQ_LEN_NO_STOP = 20
MAX_SEQ_LEN_WITH_STOP = MAX_SEQ_LEN_NO_STOP + 1 # due to the added stop token at the end

class TestModelDims(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.dataset = BCCWJDataset(indices=None, max_seq_len=MAX_SEQ_LEN_NO_STOP)
        self.dataloader = DataLoader(self.dataset, BATCH_SIZE, shuffle=True)

    def test_data_dims(self):
        some_dataset_input = next(iter(self.dataset))
        self.assertIsInstance(some_dataset_input, np.ndarray)

        some_dataloader_input = next(iter(self.dataloader))
        self.assertIsInstance(some_dataloader_input, torch.Tensor)
        self.assertEqual(some_dataloader_input.shape, (BATCH_SIZE, MAX_SEQ_LEN_WITH_STOP, NUM_PANPHON_FEATURES))

    def test_model_dims(self):
        # these are just an encoder/decoder detached from the rest of the model, made to check the dims work out
        encoder = Encoder(MAX_SEQ_LEN_WITH_STOP, NUM_PANPHON_FEATURES, HIDDEN_DIM)
        decoder = Decoder(MAX_SEQ_LEN_WITH_STOP, 2*HIDDEN_DIM, NUM_PANPHON_FEATURES) # factor of 2 comes from bidirectionality of encoder
        # an actual autoencoder, too
        autoencoder = AutoEncoder(MAX_SEQ_LEN_WITH_STOP, NUM_PANPHON_FEATURES, HIDDEN_DIM,
                                 learning_rate=1e-3,
                                 every_epoch_print=10,
                                 epochs=160,
                                 patience=5,
                                 max_grad_norm=0.005)

        some_dataloader_input = next(iter(self.dataloader))
        encoder_hidden_state = encoder(some_dataloader_input)
        # remember that we copy the hidden state as many times as we want to generate outputs,
        # hence the length is MAX_SEQ_LEN here as well
        self.assertEqual(encoder_hidden_state.shape, (BATCH_SIZE, 2*HIDDEN_DIM))

        decoder_output = decoder(encoder_hidden_state)
        self.assertEqual(decoder_output.shape, (BATCH_SIZE, MAX_SEQ_LEN_WITH_STOP, NUM_PANPHON_FEATURES))

        _encoded, autoencoder_output = autoencoder(some_dataloader_input)
        self.assertEqual(autoencoder_output.shape, (BATCH_SIZE, MAX_SEQ_LEN_WITH_STOP, NUM_PANPHON_FEATURES))


if __name__ == '__main__':
    unittest.main()