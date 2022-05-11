import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing.transcriber import Transcriber
from data_processing.bccwj_dataset import BCCWJDataset, split_pared_bccwj
from models.model import Encoder, Decoder, AutoEncoder

SEED = 888

N_WORDS = 36396

BATCH_SIZE = 64
NUM_PHONETIC_FEATURES = 22
HIDDEN_DIM = 32
MAX_SEQ_LEN_NO_STOP = 20
MAX_SEQ_LEN_WITH_STOP = MAX_SEQ_LEN_NO_STOP + 1 # due to the added stop token at the end

N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3

class TestModelDims(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.dataset = BCCWJDataset(indices=list(range(N_WORDS)), max_seq_len=MAX_SEQ_LEN_NO_STOP)
        self.dataloader = DataLoader(self.dataset, BATCH_SIZE, shuffle=True)
        self.t = Transcriber()

        # these are just an encoder/decoder detached from the rest of the model, made to check the dims work out
        self.encoder = Encoder(MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES, HIDDEN_DIM,
                               num_layers=N_ENCODER_LAYERS,
                               bidirectional=True)
        self.decoder = Decoder(MAX_SEQ_LEN_WITH_STOP, 2*HIDDEN_DIM*N_ENCODER_LAYERS, NUM_PHONETIC_FEATURES,
                               num_layers=N_DECODER_LAYERS) # factor of 2 comes from bidirectionality of encoder
        # an actual autoencoder, too. We could have tested this autoencoder's own encoder/decoder
        # but that makes it a little harder to explicitly see their parameters.
        self.autoencoder = AutoEncoder(MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES, HIDDEN_DIM,
                                       n_encoder_layers=N_ENCODER_LAYERS,
                                       n_decoder_layers=N_DECODER_LAYERS,
                                       bidirectional_encoder=True)

    def test_data_dims(self):
        some_dataset_input = next(iter(self.dataset))
        self.assertIsInstance(some_dataset_input, np.ndarray)

        some_dataloader_input = next(iter(self.dataloader))
        self.assertIsInstance(some_dataloader_input, torch.Tensor)
        self.assertEqual(some_dataloader_input.shape, (BATCH_SIZE, MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES))

    def test_model_dims(self):
        some_dataloader_input = next(iter(self.dataloader))
        encoder_hidden_state = self.encoder(some_dataloader_input)
        # remember that we copy the hidden state as many times as we want to generate outputs,
        # hence the length is MAX_SEQ_LEN here as well
        self.assertEqual(encoder_hidden_state.shape, (BATCH_SIZE, 2*HIDDEN_DIM*N_ENCODER_LAYERS))

        decoder_output = self.decoder(encoder_hidden_state)
        self.assertEqual(decoder_output.shape, (BATCH_SIZE, MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES))

        # separately of the above, check that the autoencoder pipelines things correctly
        _encoded, autoencoder_output = self.autoencoder(some_dataloader_input)
        self.assertEqual(autoencoder_output.shape, (BATCH_SIZE, MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES))

    def test_loaning_dims(self):
        some_gairaigo_fv = self.t.ipa_to_feature_vectors('kʌp')
        some_gairaigo_fv = torch.Tensor(np.array(some_gairaigo_fv))
        some_gairaigo_fv = some_gairaigo_fv.unsqueeze(0) # (1, L, H_in)

        encoder_hidden_state = self.encoder.encode(some_gairaigo_fv) # (1, 2*H_out*n_layers)
        self.assertEqual(encoder_hidden_state.shape, (1, 2*HIDDEN_DIM*N_ENCODER_LAYERS))

        decoder_output = self.decoder.decode(encoder_hidden_state) # (1, 2*L, H_in)
        self.assertEqual(decoder_output.shape, (1, 2*MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES))

        # separately of the above, check that the autoencoder pipelines things correctly
        autoencoder_output = self.autoencoder.loan_word_from_fv(some_gairaigo_fv) # (1, 2*L, H_in)
        self.assertEqual(autoencoder_output.shape, (1, 2*MAX_SEQ_LEN_WITH_STOP, NUM_PHONETIC_FEATURES))


if __name__ == '__main__':
    unittest.main()