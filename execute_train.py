import torch
from torch.utils.data import Dataset, DataLoader

from .train import train_model
from .models.model import AutoEncoder
from data_processing.bccwj_dataset import BCCWJDataset, split_pared_bccwj

SEED = 888


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

def main():
    # data
    train_dataloader, val_dataloader = split_pared_bccwj(SEED, 0.1, MAX_SEQ_LEN_WITHOUT_EOW)

    # model
    model = AutoEncoder(MAX_SEQ_LEN_WITH_EOW, N_FEATURES, HIDDEN_DIM, bidirectional=True)

    trained_model, history = train_model(model, train_dataloader, val_dataloader, device,
                                         num_epochs=num_epochs, learning_rate=learning_rate)
    torch.save(trained_model.state_dict(), './finished_checkpoint.pt')


if __name__ == '__main__':
    main()