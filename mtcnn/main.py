import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MTCNN
from data import LaSynthetic
from data import load_wv_matrix

from parser import parse_args
from logger import print_progress
from logger import print_accuracy


def train(epoch, train_loader, optimizer, criterion, train_size, args):
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

    * `train_size`: [int]
        Size of the training set (for logging).

    * `args`: [argparse object]
        Parsed arguments.
    """
    model.train()
    for batch_idx, sample in enumerate(train_loader):
#        print('input data size: {}'.format(sample['sentence'].size()))
        sentence = sample['sentence']
        subsite = sample['subsite']
        laterality = sample['laterality']
        behavior = sample['behavior']
        histology = sample['histology']
        grade = sample['grade']

        if args.cuda:
            sentence = sentence.cuda()
            subsite = subsite.cuda()
            laterality = laterality.cuda()
            behavior = behavior.cuda()
            histology = histology.cuda()
            grade = grade.cuda()

        sentence = Variable(sentence)
        subsite = Variable(subsite)
        laterality = Variable(laterality)
        behavior = Variable(behavior)
        histology = Variable(histology)
        grade = Variable(grade)

        optimizer.zero_grad()
        out_subsite, out_laterality, out_behavior, out_histology, out_grade = model(sentence)
        loss_subsite = criterion(out_subsite, subsite)
        loss_laterality = criterion(out_laterality, laterality)
        loss_behavior = criterion(out_behavior, behavior)
        loss_histology = criterion(out_histology, histology)
        loss_grade = criterion(out_grade, grade)
        loss = loss_subsite + loss_laterality + loss_behavior + loss_histology + loss_grade
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_progress(epoch, batch_idx, args.batch_size, train_size, loss.data[0])


def test(epoch, test_loader, args):
    """
    Test the model.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader for the test set.

    * `args`: [argparse object]
        Parsed arguments.
    """
    model.eval()
    subsite_correct = 0
    laterality_correct = 0
    behavior_correct = 0
    grade_correct = 0
    total = 0

    for _, sample in enumerate(test_loader):
        sentence = sample['sentence']
        subsite = sample['subsite']
        laterality = sample['laterality']
        behavior = sample['behavior']
        histology = sample['histology']
        grade = sample['grade']

        if args.cuda:
            sentence = sentence.cuda()
            subsite = subsite.cuda()
            laterality = laterality.cuda()
            behavior = behavior.cuda()
            histology = histology.cuda()
            grade = grade.cuda()

        sentence = Variable(sentence)
        subsite = Variable(subsite)
        laterality = Variable(laterality)
        behavior = Variable(behavior)
        histology = Variable(histology)
        grade = Variable(grade)

        out_subsite, out_laterality, out_behavior, out_histology, out_grade = model(sentence)
        _, subsite_predicted = torch.max(out_subsite.data, 1)
        _, laterality_predicted = torch.max(out_laterality.data, 1)
        _, behavior_predicted = torch.max(out_behavior.data, 1)
        _, histology_predicted = torch.max(out_histology.data, 1)
        _, grade_predicted = torch.max(out_grade.data, 1)

        total += subsite.size(0)
        subsite_correct += (subsite_predicted == subsite.data).sum()
        laterality_correct += (laterality_predicted == laterality.data).sum()
        behavior_correct += (behavior_predicted == behavior.data).sum()
        grade_correct += (grade_predicted == grade.data).cpu().sum()

    print_accuracy(
        epoch, subsite_correct, laterality_correct,
        behavior_correct, histology_correct, grade_correct, total
    )


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_data = LaSynthetic(
        data_path=args.data_dir + '/data/train',
        label_path=args.data_dir + '/labels/train'
    )

    train_size = len(train_data)

    test_data = LaSynthetic(
        data_path=args.data_dir + '/data/val',
        label_path=args.data_dir + '/labels/val'
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    global model
    model = MTCNN(kernel1=3, kernel2=4, kernel3=5)
    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        if args.half_precision:
            model.half()

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.num_epochs + 1):
            train(epoch, train_loader, optimizer, criterion, train_size, args)
            test(epoch, test_loader, args)


if __name__=='__main__':
    main()
