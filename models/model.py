'''
Code for the autoencoder model.

Largely based off of these resources:
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html for the Encoder/Decoder code
- https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py for the Encoder/Decoder code and EarlyStopping
'''

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

SEED = 888
torch.manual_seed(SEED)

# TODO for now only supports bidirectional, all the code assumes it

class Encoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.):
        # input_size is the number of features in the input representation
        # hidden_size is the dimension of the hidden state
        # remember that the hidden state must capture not just a single input but the entire
        # sequential input, so hidden_size > input_size is both reasonable and likely.
        super().__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout)
    
    def forward(self, x):
        # x: (N, L, H_in)
        # convert type of x tensor to float for use with RNN
        x = x.type(torch.FloatTensor)
        x, h_n = self.rnn(x) # x: (N, L, 2*H_out)
                             # h_n: (2*n_layers, N, H_out)
        # concatenate the layers' hidden representations
        fwd_final_h_n = h_n[0:h_n.size(0):2, :, :] # (n_layers, N, H_out)
        bwd_final_h_n = h_n[1:h_n.size(0):2, :, :] # (n_layers, N, H_out)
        x = torch.cat([fwd_final_h_n, bwd_final_h_n], dim=2) # (n_layers, N, 2*H_out)
        x = torch.permute(x, (1,0,2)) # (N, n_layers, 2*H_out)
        # rearrange the dimensions to make it more natural, simply associating each input with a single concatenated hidden state
        x = torch.reshape(x, (x.size(0), -1)) # (N, 2*H_out*n_layers)
        return x

    def encode(self, x):
        # lets one use the model as an encoder alone, without training.
        # x: (N, L, H_in)
        self.eval()
        with torch.no_grad():
            x = x.type(torch.FloatTensor)
            x, h_n = self.rnn(x) # x: (N, L, 2*H_out)
                                 # h_n: (2*n_layers, N, H_out)
        # concatenate the layers' hidden representations
        fwd_final_h_n = h_n[0:h_n.size(0):2, :, :] # (n_layers, N, H_out)
        bwd_final_h_n = h_n[1:h_n.size(0):2, :, :] # (n_layers, N, H_out)
        x = torch.cat([fwd_final_h_n, bwd_final_h_n], dim=2) # (n_layers, N, 2*H_out)
        x = torch.permute(x, (1,0,2)) # (N, n_layers, 2*H_out)
        x = torch.reshape(x, (x.size(0), -1)) # (N, 2*H_out*n_layers)
        self.train()
        return x


class Decoder(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, num_layers=1, dropout=0.):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(hidden_size, output_size, num_layers,
                           batch_first=True,
                           dropout=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, H_in)
        # we need to copy the hidden state tensor L times for the L decodes:
        x = x.unsqueeze(1) # (N, 1, H_in)
        x = x.repeat(1, self.seq_len, 1) # (N, L, H_in)
        x, h_n = self.rnn(x) # x: (N, L, H_out)
                             # h_n: (n_layers, N, H_out)
        x = self.sigmoid(x) # x: (N, L, H_out)
        return x

    def decode(self, x):
        # lets one use the model as a decoder alone, without training it.
        # when decoding, we let the model potentially predict sequences twice as long
        # as the input. Hence, we concatenate x with itself to get a 2L long sequence.
        # x: (N, H_in) nb that in principle one can use this to decode many words at once, even though we typically only do one
        self.eval()
        with torch.no_grad():
            x = x.unsqueeze(1) # (N, 1, H_in)
            x = x.repeat(1, 2*self.seq_len, 1) # (N, 2*L, H_in)
            x, h_n = self.rnn(x) # x: (N, 2*L, H_out)
                                 # h_n: (n_layers, N, H_out)
            x = self.sigmoid(x)
        self.train()
        return x


class AutoEncoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size,
                 n_encoder_layers=3,
                 n_decoder_layers=3,
                 bidirectional_encoder=True,
                 enc_dropout=0.1,
                 dec_dropout=0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        enc_directions = 2 if bidirectional_encoder else 1
        decoder_input_factor = enc_directions * n_encoder_layers

        self.encoder = Encoder(self.seq_len, self.input_size, self.hidden_size,
                               num_layers=n_encoder_layers,
                               bidirectional=bidirectional_encoder,
                               dropout=enc_dropout)
        self.decoder = Decoder(self.seq_len, (decoder_input_factor * self.hidden_size), self.input_size,
                               num_layers=n_decoder_layers,
                               dropout=dec_dropout)

    def forward(self, x):
        # x: (N, L, H_in) ie (N, L, input_size)
        encoded = self.encoder(x) # encoded: (N, 2*H_out*n_layers) ie (N, 2*hidden_size*n_layers)
        decoded = self.decoder(encoded) # decoded: (N, L, H_out) ie (N, L, input_size)
        return encoded, decoded

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded

    def decode(self, encoded):
        self.eval()
        with torch.no_grad():
            decoded = self.decoder(encoded)
            squeezed_decoded = decoded.squeeze() # as the output will often have N=1 anyway
        return squeezed_decoded
    
    def loan_word_from_fv(self, x):
        # x: (N, L, H_in) ; N=1 typically but in principle you could batch things
        # wraps the encode/decode process. Doesn't train the model.
        self.eval()
        with torch.no_grad():
            encoded = self.encoder.encode(x) # (N, 2*H_out)
            decoded = self.decoder.decode(encoded) # (N, 2*L, H_in)
        self.train()
        return decoded

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))
