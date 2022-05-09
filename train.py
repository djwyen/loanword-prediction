import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import AutoEncoder
from data_processing.bccwj_dataset import split_pared_bccwj

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
parameters = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 2}
num_epochs = 50
learning_rate = 1e-3

# parameters for splitting up the dataset
random_seed = 888
frac_test = 0.1
printout_freq = 1500 # print every 1500th word during training; should give 24 printouts

N_FEATURES = 24 # from how panphon works by default
HIDDEN_DIM = 32 # remember that the hidden state must contain the entire sequence, somehow
MAX_SEQ_LEN_WITHOUT_EOW = 20
MAX_SEQ_LEN_WITH_EOW = MAX_SEQ_LEN_WITHOUT_EOW + 1


# largely based off of https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
def train_model(model, train_dataloader, val_dataloader, device,
                num_epochs=50, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    history = dict(train=[], val=[])

    best_loss = None
    for epoch in tqdm(range(1, num_epochs+1)):
        # train
        train_losses = []
        model.train()
        for target in train_dataloader:
            optimizer.zero_grad()
            target = target.to(device)
            _encoded, prediction = model(target)

            target = target.type(torch.FloatTensor) # to allow loss comparison
            loss = criterion(prediction, target)

            # backward pass
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # evaluate
        val_losses = []
        model.eval()
        with torch.no_grad():
            for target in val_dataloader:
                target = target.to(device)
                _encoded, prediction = model(target)

                target = target.type(torch.FloatTensor)
                loss = criterion(prediction, target)
                val_losses.append(loss.item())
        
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './checkpoint.pt')

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f'Epoch : {epoch}, train_loss: {train_loss:.7f}, val_loss: {val_loss:.7f}')
    
    model.load_state_dict(torch.load('./checkpoint.pt'))
    model.eval()

    return model, history


def main():
    # data
    train_dataset, val_dataset = split_pared_bccwj(random_seed, 0.1, MAX_SEQ_LEN_WITHOUT_EOW)

    train_dataloader = DataLoader(train_dataset,
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'])
    val_dataloader = DataLoader(val_dataset,
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'])


    # model
    model = AutoEncoder(MAX_SEQ_LEN_WITH_EOW, N_FEATURES, HIDDEN_DIM, bidirectional=True)

    trained_model, history = train_model(model, train_dataloader, val_dataloader, device,
                                         num_epochs=num_epochs, learning_rate=learning_rate)
    torch.save(trained_model.state_dict(), './finished_checkpoint.pt')


if __name__ == '__main__':
    main()