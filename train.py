import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import AutoEncoder
from data_processing.bccwj_dataset import split_pared_bccwj
from process_bccwj_to_fv import NUM_PHONETIC_FEATURES, CATEGORIES_PER_FEATURE

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
parameters = {'batch_size': 64,
              'shuffle': True,
              'num_epochs': 100,
              'learning_rate': 1e-3,
              'encoder_layers': 3,
              'decoder_layers': 3,
              'encoder_dropout': 0.1,
              'decoder_dropout': 0.3}

# parameters for splitting up the dataset
SEED = 888
frac_test = 0.1
printout_freq = 1500 # print every 1500th word during training; should give 24 printouts

HIDDEN_DIM = 32 # remember that the hidden state must contain the entire sequence, somehow
MAX_SEQ_LEN_WITHOUT_EOW = 20
MAX_SEQ_LEN_WITH_EOW = MAX_SEQ_LEN_WITHOUT_EOW + 1

# pulled from https://github.com/dmort27/panphon/blob/master/panphon/data/feature_weights.csv
# some arbitrary weighting of the features
FEATURE_WEIGHTS = [1,1,1,0.5,0.25,0.25,0.25,0.125,0.125,0.125,0.125,0.25,0.25,0.125,0.25,0.25,0.25,0.25,0.25,0.25,0.125,0.25]

def weighted_loss(prediction, target):
    '''DEPRECATED after change to how feature vectors are represented; this will no longer work out of the box'''
    weight_tensor = np.array(FEATURE_WEIGHTS)
    weight_tensor = torch.tensor(weight_tensor).unsqueeze(0) # (1, H_in)
    repeated_weights = weight_tensor.expand((target.shape[0], MAX_SEQ_LEN_WITH_EOW, -1)) # (N, L, H_in)

    # prediction = torch.mul(repeated_weights, prediction)
    # target = torch.mul(repeated_weights, target)

    # the factor of 1000 is arbitrary to keep numbers larger
    loss = 1000*torch.mean(repeated_weights * (target - prediction)**2)
    return loss

def masked_bce(pred, target, tgt_lengths):
    # pred: (N, L, H_out)
    # target: (N, L, H_out)
    # tgt_lengths: (N,)
    
    # flatten the time-stepped predictions/target into single vectors
    pred = torch.reshape(pred, (pred.size(0), -1)) # (N, L*H_out)
    target = torch.reshape(target, (target.size(0), -1)) # (N, L*H_out)

    # calculate the cross-entropy on _all_ values. Notably we do not apply the weight at this time
    cross_entropy_vec = F.binary_cross_entropy_with_logits(pred, target) # (N, L*H_out)

    # construct the weight vector to pointwise multiply with the cross entropy vector
    # this has the double purpose of applying a weighting to particular features
    # and letting us zero out the loss from PAD tokens
    weight_vec = torch.tensor(np.array(FEATURE_WEIGHTS)) # (H_out,)
    iterated_weight_vec = weight_vec.repeat((pred.size(0), MAX_SEQ_LEN_WITH_EOW)) # (N, L*H_out)
    # ngl I don't understand how this works, pulled from https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    # it creates an array such that the ith element is True iff the vector at that position should
    # be retained and not masked away from the loss
    mask_vec = torch.arange(MAX_SEQ_LEN_WITH_EOW).expand(len(tgt_lengths), MAX_SEQ_LEN_WITH_EOW) < tgt_lengths.unsqueeze(1) # (N, L)
    mask_vec = mask_vec.unsqueeze(2) # (N, L, 1)
    mask_vec = mask_vec.repeat((1, 1, NUM_PHONETIC_FEATURES)) # (N, L, H_out)
    mask_vec = torch.reshape(mask_vec, (mask_vec.size(0), -1)) # (N, L*H_out)
    weight_mask_vec = iterated_weight_vec * mask_vec # (N, L*H_out)
    weighted_masked_crossent = cross_entropy_vec * weight_mask_vec # (N, L*H_out)
    return torch.sum(weighted_masked_crossent) / pred.size(0) # take average loss per entry

# largely based off of https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
def train_model(model, train_dataloader, val_dataloader, device,
                num_epochs=100, learning_rate=1e-3, print_every_n_epochs=25):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # triplicate_feature_weights = []
    # for x in FEATURE_WEIGHTS:
    #     triplicate_feature_weights.extend([x, x, x])
    weights_tensor = torch.tensor(np.array(FEATURE_WEIGHTS)).type(torch.FloatTensor)
    criterion = nn.BCEWithLogitsLoss(weight=weights_tensor, reduction='sum')
    
    history = dict(train=[], val=[])

    best_loss = None
    for epoch in tqdm(range(1, num_epochs+1)):
        # train
        train_losses = []
        model.train()
        for target, tgt_lengths in train_dataloader:
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
            for i, (target, tgt_lengths) in enumerate(val_dataloader):
                target = target.to(device)
                _encoded, prediction = model(target)

                target = target.type(torch.FloatTensor)
                loss = criterion(prediction, target)
                val_losses.append(loss.item())

                if (i == 0) and (epoch % print_every_n_epochs == 0):
                    print('---')
                    print('fv of first segment of first word in validation batch:')
                    print('prediction:')
                    print(F.sigmoid(prediction).numpy()[0, 0, :])
                    print('target:')
                    print(target.numpy()[0, 0, :])
                    print('---')

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f'Epoch : {epoch}, train_loss: {train_loss:.7f}, val_loss: {val_loss:.7f}')
        
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './checkpoint.pt')
    
    model.load_state_dict(torch.load('./checkpoint.pt'))
    model.eval()

    return model, history


def main():
    # data
    train_dataset, val_dataset = split_pared_bccwj(SEED, frac_test, MAX_SEQ_LEN_WITHOUT_EOW)

    train_dataloader = DataLoader(train_dataset,
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'])
    val_dataloader = DataLoader(val_dataset,
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'])


    # model
    model = AutoEncoder(MAX_SEQ_LEN_WITH_EOW, NUM_PHONETIC_FEATURES, HIDDEN_DIM,
                        n_encoder_layers=parameters['encoder_layers'],
                        n_decoder_layers=parameters['decoder_layers'],
                        bidirectional_encoder=True,
                        enc_dropout=parameters['encoder_dropout'],
                        dec_dropout=parameters['decoder_dropout'])

    trained_model, history = train_model(model, train_dataloader, val_dataloader, device,
                                         num_epochs=parameters['num_epochs'],
                                         learning_rate=parameters['learning_rate'])
    torch.save(trained_model.state_dict(), './finished_checkpoint.pt')


if __name__ == '__main__':
    main()
