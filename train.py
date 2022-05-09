import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

# largely based off of https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
def train_model(model, train_dataloader, val_dataloader, device,
                num_epochs=50, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    history = dict(train=[], val=[])

    best_loss = None
    for epoch in tqdm(range(1, model.epochs+1)):
        # early_stopping.epoch = epoch

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
