
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/model_1')

from sklearn.preprocessing import StandardScaler




parser = argparse.ArgumentParser(description='PyTorch FraudDetection')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)


#device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################
# Simple Linear Model
######################################################

class FraudNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 18)
        self.fc3 = nn.Linear(18, 20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x




########################################################
# Load data
########################################################
"""

#x_path = str(TRAIN) + '/X_train.csv'
#y_path = str(TRAIN) + '/y_train.csv'
df_train = pd.read_csv('../dataset/train/df_train.csv')

df_x = df_train.iloc[:, :-1]
df_y = df_train.pop('Class')


print('dataset loaded')
X_train = np.asarray(df_x.values)
y_train = np.asarray(df_y.values)
y_train = y_train.reshape(y_train.shape[0], 1)
print(f'Before: X_train.shape = {X_train.shape}, y_train.shape: {y_train.shape}')


sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)


# Train
net = FraudNet().double().to(device)


X_train_sc = torch.from_numpy(X_train_sc).double().to(device)
y_train = torch.from_numpy(y_train).double().to(device)
print('after torh.from_numpy')
print(f'X_train.shape = {X_train.shape}, y_train.shape: {y_train.shape}')

# Binary-Cross-Error loss (it requires the sigmoid's output as input)
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

training_epochs = 5
minibatch_size = 64

train = data_utils.TensorDataset(X_train_sc, y_train)
train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True)
# add network graph to tensorboard
print(f'Adding network: X_train_sc[0].shape: {X_train_sc[0].shape}')
writer.add_graph(net, X_train_sc[0])
writer.close()


############################################
# training the model
############################################
running_loss = 0.0
for i in range(training_epochs):
    for b, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.reshape(labels.shape[0], 1)
        y_pred = net(inputs)

        loss = criterion(y_pred, labels)  # y_pred, labels must be the same shape.
        running_loss += loss.item()
        if b % 1000 == 999:  # every 1000 mini-batches
            print(f'Epochs: {i}, batch: {b} loss: {loss.item()}')
            # update tensorboard
            writer.add_scalar('Train/Loss', running_loss / 1000, i + 1 + b/1000)
            writer.flush()


        # reset gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    # update tensorboard
    writer.add_scalar('Train/Loss', loss.item(), i+1)
    writer.flush()



 # Save the model
checkpoint = {'model': FraudNet(),
          'state_dict': net.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'models/checkpoint_1.pth')
"""


"""
#################################################################################
# Test validation
#################################################################################
X_test = torch.from_numpy(X_test).double().to(device)
y_test = torch.from_numpy(y_test).double().to(device)

test = data_utils.TensorDataset(X_test, y_test)
test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.double() == labels).sum().item()

print(f'Accuracy of the network on the {X_test.shape[0]} inputs: {100*correct/total}%')
"""