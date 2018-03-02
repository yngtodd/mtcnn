import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from parser import parse_args

from data import Deidentified
from model import MTCNN


def train(epoch, train_loader, model, optimizer, args):
    """
    Train the model.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.

    * `train_loader` [torch.utils.data.Dataloader]
        Data loader to load the test set.

    * `model`: [Pytorch model class]
        Instantiated model.

    * `optimizer`: [torch.optim optimizer]
        Optimizer for learning the model.

    * `args`: [argparse object]
        Parsed arguments.
    """
    model.train()
    for batch_idx, sample in enumerate(train_loader):



def test(test_loader, model, args):
    """
    Test the model.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader for the test set.

    * `model`: [Pytorch model class]
        Instantiated model.

    * `args`: [argparse object]
        Parsed arguments.
    """


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_data = Deidentified(
        data_path=args.data_dir + '/data/train',
        label_path=args.data_dir + '/labels/train'
    )

    test_data = Deidentified(
        data_path=args.data_dir + '/data/test',
        label_path=args.data_dir + '/labels/test'
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = MTCNN()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.num_epochs + 1):
            train(epoch, train_loader, model, optimizer, args)
            test(test_loader, model, args)
