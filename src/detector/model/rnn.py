import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn

parser = argparse.ArgumentParser(description='PyTorch FraudDetection')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

####################################################################
# Load Data
####################################################################


class Net(nn.Module):
    '''
    Very simple network consisting of an embedding layer, LSTM layers and a decoder with dropouts
    '''

    def __init__(self, n_actions=6, embedding_size=3, n_nodes=6, n_layers=2, dropout=0.2,
                 padding_idx=0, initrange=0.5):
        super(VerySimpleBehaviorNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(n_actions, embedding_size, padding_idx)
        self.rnn = nn.LSTM(embedding_size, n_nodes, n_layers, dropout=dropout)
        self.decoder = nn.Linear(n_nodes, n_actions)
        self.init_weights(initrange)
        self.n_nodes = n_nodes
        self.n_layers = n_layers

    def init_weights(self, initrange=0.1):
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Set the first row to zero (padding idx)
        self.embedding.weight.data[0, :] = 0
        print(self.embedding.weight)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.n_nodes),
                weight.new_zeros(self.n_layers, batch_size, self.n_nodes))

    def forward(self, input, hidden):
        emb = self.dropout(self.embedding(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
