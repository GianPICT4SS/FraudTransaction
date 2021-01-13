import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/model_0')


from utils.fraud_net import FraudNet
from utils.config import MODELS
from utils.messages_utils import publish_training_completed

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model, train_rdd, y_rdd, model_id, device):
    """train a new model with a dataset bigger than before and save it on path MODELS/checkpoint_model_id.pth
    """
    X_train_sc = torch.from_numpy(train_rdd).double().to(device)
    y_train = torch.from_numpy(y_rdd).double().to(device)
    # Binary-Cross-Error loss (it requires the sigmoid's output as input)
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_epochs = 5
    minibatch_size = 64

    train_dataset = data_utils.TensorDataset(X_train_sc, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)
    # add network graph to tensorboard
    print(f'Adding network: X_train_sc[0].shape: {X_train_sc[0].shape}')
    writer.add_graph(model, X_train_sc[0])
    writer.close()

    ############################################
    # training the model
    ############################################
    running_loss = 0.0
    for i in range(training_epochs):
        for b, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.reshape(labels.shape[0], 1)
            y_pred = model(inputs)
            _, predicted = torch.max(y_pred.data, 1)
            if predicted.sum().item() != 0:
                print(f'fraud: {predicted.float()}')

            loss = criterion(y_pred, labels)  # y_pred, labels must be the same shape.
            running_loss += loss.item()
            if b % 1000 == 999:  # every 1000 mini-batches
                print(f'Epochs: {i}, batch: {b} loss: {loss.item()}')
                # update tensorboard
                writer.add_scalar('Train/Loss', running_loss / 1000, i + 1 + b / 1000)
                writer.flush()

            # reset gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

        # update tensorboard
        writer.add_scalar('Train/Loss', loss.item(), i + 1)
        writer.flush()

    # Save the model
    checkpoint = {'model': FraudNet(),
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    chechpoint_path = f'./model/models/{model_id}.pth'
    torch.save(checkpoint, chechpoint_path)
    logger.info('Training phase finisched.')
    publish_training_completed(model_id)