import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .data_processing.bccwj_dataset import BCCWJDataset, split_pared_bccwj
from .models import AutoEncoder
from .models.early_stopping import EarlyStopping

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
parameters = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
num_epochs = 100

# parameters for splitting up the dataset
random_seed = 888
frac_test = 0.1
printout_freq = 1500 # print every 1500th word during training; should give 24 printouts

N_FEATURES = 24 # from how panphon works by default
HIDDEN_DIM = 32 # remember that the hidden state must contain the entire sequence, somehow

# Datasets

train_dataset, test_dataset = split_pared_bccwj(random_seed, frac_test)

train_dataloader = DataLoader(train_dataset,
                             batch_size=parameters['batch_size'],
                             shuffle=parameters['shuffle'],
                             num_workers=parameters['num_workers'])
test_dataloader = DataLoader(test_dataset,
                             batch_size=parameters['batch_size'],
                             shuffle=parameters['shuffle'],
                             num_workers=parameters['num_workers'])

# TODO refactor so the parameters are more sensibly accessed...
def train_model(model, train_dataloader, val_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr = model.learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    history = dict(train=[], val=[])

    early_stopping = EarlyStopping(patience=model.patience, verbose=False)

    for epoch in tqdm(range(1, model.epochs+1)):
        early_stopping.epoch = epoch

        train_losses = []
        model.train()
        for target in train_dataloader:
            optimizer.zero_grad()
            target = target.to(device)
            _encoded, prediction = model(target)

            loss = criterion(prediction, target)

            # check if we should stop early due to no improvement
            early_stopping(loss, model)
            if early_stopping.early_stop:
                break

            # backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=model.max_grad_norm)
            optimizer.step()

            train_losses.append(loss.item())
        
        # evaluate
        val_losses = []
        model.eval()
        with torch.no_grad():
            for target in val_dataloader:
                target = target.to(device)
                _encoded, prediction = model(target)

                loss = criterion(prediction, target)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f'Epoch : {epoch}, train_loss: {train_loss:.7f}, val_loss: {val_loss:.7f}')
    
    model.load_state_dict(torch.load('./checkpoint.pt'))

    return model.eval(), history
