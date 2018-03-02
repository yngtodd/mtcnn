import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from parser import parse_args

from data import Deidentified


def train(epoch, train_loader):
    """
    Train the model.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.

    * `train_loader` [torch.utils.data.Dataloader]
        Data loader to load the test set.
    """


def test(test_loader):
    """
    Test the model.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader for the test set.
    """


def main():
    args = parse_args()

    train_data = Deidentified(data_path=args.data_dir+'/data/train', label_path=args.data_dir+'/labels/train')
    test_data = Deidentified(data_path=args.data_dir+'/data/test', label_path=args.data_dir+'/labels/test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    for epoch in range(1, args.num_epochs + 1):
            train(epoch, train_loader)
            test(test_loader)
