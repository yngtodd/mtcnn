import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MTCNN
from data import Deidentified
from data import load_wv_matrix

from parser import parse_args


def train(epoch, train_loader, model, optimizer, criterion, args):
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

    * `criterion`: [torch loss function]
        Loss function to measure learning.

    * `args`: [argparse object]
        Parsed arguments.
    """
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        sentence = sample['sentence']
        subsite = sample['subsite']
        laterality = sample['laterality']
        behavior = sample['behavior']
        grade = sample['grade']

        sentence = Variable(sentence)
        subsite = Variable(subsite)
        laterality = Variable(laterality)
        behavior = Variable(behavior)
        grade = Variable(grade)
#        print('sentence is of type {} and shape {}'.format(type(sentence), sentence.size()))

        optimizer.zero_grad()
        out_subsite, out_laterality, out_behavior, out_grade = model(sentence)

        loss_subsite = criterion(out_subsite, subsite)
        loss_laterality = criterion(out_laterality, laterality)
        loss_behavior = criterion(out_behavior, behavior)
        loss_grade = criterion(out_grade, grade)
        loss = loss_subsite + loss_laterality + loss_behavior + loss_grade
        loss.backward()
        optimizer.step()

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
    print('parsing args')
    args = parse_args()
    print(args)
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
    wv_matrix = load_wv_matrix(args.data_dir + '/wv_matrix/wv_matrix.npy')

    model = MTCNN(wv_matrix)
    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.num_epochs + 1):
            train(epoch, train_loader, model, optimizer, criterion, args)
            test(test_loader, model, args)


if __name__=='__main__':
    main()
