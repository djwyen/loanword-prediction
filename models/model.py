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
from .early_stopping import *

SEED = 888
torch.manual_seed(SEED)

class Token:
    PAD = 0 # should it be something else than all zeroes?
    SOS = 2
    EOS = -2

# TODO for now only supports bidirectional, all the code assumes it
# TODO the dropout is only applied between layers so for 1 layer rnns this does nothing

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
        # self.softmax = nn.LogSoftmax(dim = 1)

        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
        #                   batch_first=True,
        #                   bidirectional=bidirectional,
        #                   dropout=dropout)
    
    def forward(self, x):
        # x: (N, L, H_in)
        # convert type of x tensor to float for use with RNN
        x = x.type(torch.FloatTensor)
        x, h_n = self.rnn(x) # x: (N, L, 2H_out)
                             # h_n: (2, N, H_out)
        # concatenate the layers' hidden representations
        fwd_final_h_n = h_n[0, :, :] # (N, H_out)
        bwd_final_h_n = h_n[1, :, :] # (N, H_out)
        x = torch.cat([fwd_final_h_n, bwd_final_h_n], dim=1) # (N, 2H_out)
        return x

    def encode(self, x):
        # lets one use the model as an encoder alone, without training.
        # x: (N, L, H_in)
        self.eval()
        with torch.no_grad():
            x, h_n = self.rnn(x) # x: (N, L, 2H_out)
                                 # h_n: (2, N, H_out)
        # concatenate the layers' hidden representations
        fwd_final_h_n = h_n[0, :, :] # (N, H_out)
        bwd_final_h_n = h_n[1, :, :] # (N, H_out)
        x = torch.cat([fwd_final_h_n, bwd_final_h_n], dim=1) # (N, 2H_out)
        self.train()
        return x


class Decoder(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, num_layers=1, dropout=0.5):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(hidden_size, output_size, num_layers,
                           batch_first=True,
                           dropout=dropout)

    def forward(self, x):
        # x: (N, H_in)
        # we need to copy the tensor L times for the L decodes:
        x = x.unsqueeze(1) # (N, 1, 2H_out)
        x = x.repeat(1, self.seq_len, 1) # (N, L, H_in)
        x, h_n = self.rnn(x) # x: (N, L, H_out)
                             # h_n: (1, N, H_out)
        return x

    def decode(self, x):
        # lets one use the model as a decoder alone, without training it.
        # when decoding, we let the model potentially predict sequences twice as long
        # as the input. Hence, we concatenate x with itself to get a 2L long sequence.
        # x: (N, H_in) nb that in principle one can use this to decode many words at once, even though we typically only do one
        x = x.unsqueeze(1)
        x = x.repeat(1, 2*self.seq_len, 1)
        self.eval()
        with torch.no_grad():
            x, h_n = self.rnn(x) # x: (N, 2L, H_out)
                                 # h_n: (1, N, H_out)
        self.train()
        return x


class AutoEncoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size,
                 learning_rate, epochs, max_grad_norm,
                 bidirectional=True,
                 patience=20, every_epoch_print=10):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        directions = 2 if bidirectional else 1

        self.encoder = Encoder(self.seq_len, self.input_size, self.hidden_size, bidirectional=bidirectional)
        self.decoder = Decoder(self.seq_len, (directions * self.hidden_size), self.input_size)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.every_epoch_print = every_epoch_print

    def forward(self, x):
        # x: (N, L, H_in) ie (N, L, input_size)
        torch.manual_seed(SEED)
        encoded = self.encoder(x) # encoded: (N, 1, 2H_out) ie (N, 1, 2hidden_size)
        decoded = self.decoder(encoded) # decoded: (N, L, H_out) ie (N, L, input_size)
        return encoded, decoded

    def fit(self, x):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        criterion = nn.MSELoss(reduction='mean')
        self.train()

        early_stopping = EarlyStopping(patience=self.patience, verbose=False)

        for epoch in range (1, self.epochs+1):
            early_stopping.epoch = epoch
            optimizer.zero_grad()
            encoded, decoded = self(x)
            loss = criterion(decoded, x)

            early_stopping(loss, self)

            if early_stopping.early_stop:
                break

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            optimizer.step()

            if epoch % self.every_epoch_print == 0:
                print(f'epoch : {epoch}, loss_mean : {loss.item():.7f}')
        
        self.load_state_dict(torch.load('./checkpoint.pt'))

        encoded, decoded = self(x)
        final_loss = criterion(decoded, x).item()

        return final_loss

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded

    def decode(self, encoded):
        self.eval()
        with torch.no_grad():
            decoded = self.decoder(encoded)
            squeezed_decoded = decoded.squeeze()
        return squeezed_decoded
    
    def loan_word_from_fv(self, x):
        # x: (N, L, H_in) ; N=1 typically but in principle you could batch things
        # wraps the encode/decode process. Doesn't train the model.
        self.eval()
        with torch.no_grad():
            encoded = self.encoder.encode(x) # (1, 2H_out)
            decoded = self.decoder.decode(encoded) # (1, 2L, H_in)
        self.train()
        return decoded

    def load(self, PATH):
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))
