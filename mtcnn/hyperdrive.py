import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MTCNN
from data import Deidentified
from data import load_wv_matrix

from parser import parse_args
from logger import print_accuracy

from hyperspace import hyperdrive


def objective(hparams):
    """
    Train the model.

    Parameters:
    ----------
    * `hparams`: [list]
        Hyperparameters set by HyperSpace.

    * `train_loader` [torch.utils.data.Dataloader]
        Data loader to load the test set.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader to load the validation set.

    * `optimizer`: [torch.optim optimizer]
        Optimizer for learning the model.

    * `criterion`: [torch loss function]
        Loss function to measure learning.

    * `train_size`: [int]
        Size of the training set (for logging).

    * `args`: [argparse object]
        Parsed arguments.
    """
    kernel1, kernel2, kernel3, num_filters1, num_filters2, num_filters3 = hparams

    global optimizer
    optimizer = optim.Adam(model.parameters())

    global model
    model = MTCNN(
        wv_matrix, kernel1=kernel1, kernel2=kernel2, kernel3, kernel3,
        num_filters1, num_filters2, num_filters3
    )

    if args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    model.train()
    for epoch in range(1, args.num_epochs + 1):
            for batch_idx, sample in enumerate(train_loader):
                sentence = sample['sentence']
                subsite = sample['subsite']
                laterality = sample['laterality']
                behavior = sample['behavior']
                grade = sample['grade']

                if args.cuda:
                    sentence = sentence.cuda()
                    subsite = subsite.cuda()
                    laterality = laterality.cuda()
                    behavior = behavior.cuda()
                    grade = grade.cuda()
                    if args.half_precision:
                        sentence = sentence.half()
                        subsite = subsite.half()
                        laterality = laterality.half()
                        behavior = behavior.half()
                        grade = grade.half()

                sentence = Variable(sentence)
                subsite = Variable(subsite)
                laterality = Variable(laterality)
                behavior = Variable(behavior)
                grade = Variable(grade)

                optimizer.zero_grad()
                out_subsite, out_laterality, out_behavior, out_grade = model(sentence)
                loss_subsite = criterion(out_subsite, subsite)
                loss_laterality = criterion(out_laterality, laterality)
                loss_behavior = criterion(out_behavior, behavior)
                loss_grade = criterion(out_grade, grade)
                loss = loss_subsite + loss_laterality + loss_behavior + loss_grade
                loss.backward()
                optimizer.step()

    ave_loss = validate(test_loader, criterion, args)
    return ave_loss


def validate(test_loader, criterion, args):
    """
    Validate the model.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.

    * `test_loader`: [torch.utils.data.Dataloader]
        Data loader for the test set.

    * `criterion`: [torch loss function]
        Loss function to measure learning.

    * `args`: [argparse object]
        Parsed arguments.

    Returns:
    -------
    ave_loss: [float]
        Average loss on the validation set
    """
    model.eval()
    loss = 0
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
        grade = sample['grade']

        if args.cuda:
            sentence = sentence.cuda()
            subsite = subsite.cuda()
            laterality = laterality.cuda()
            behavior = behavior.cuda()
            grade = grade.cuda()

        sentence = Variable(sentence)
        subsite = Variable(subsite)
        laterality = Variable(laterality)
        behavior = Variable(behavior)
        grade = Variable(grade)

        out_subsite, out_laterality, out_behavior, out_grade = model(sentence)
        loss_subsite = criterion(out_subsite, subsite)
        loss_laterality = criterion(out_laterality, laterality)
        loss_behavior = criterion(out_behavior, behavior)
        loss_grade = criterion(out_grade, grade)
        loss += loss_subsite + loss_laterality + loss_behavior + loss_grade

        _, subsite_predicted = torch.max(out_subsite.data, 1)
        _, laterality_predicted = torch.max(out_laterality.data, 1)
        _, behavior_predicted = torch.max(out_behavior.data, 1)
        _, grade_predicted = torch.max(out_grade.data, 1)

        total += subsite.size(0)
        subsite_correct += (subsite_predicted == subsite).sum()
        laterality_correct += (laterality_predicted == laterality).sum()
        behavior_correct += (behavior_predicted == behavior).sum()
        grade_correct += (grade_predicted == grade).sum()

    print_accuracy(
        subsite_correct, laterality_correct,
        behavior_correct, grade_correct, total
    )

    ave_loss = loss / len(test_loader)
    return ave_loss


def main():
    global args
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_data = Deidentified(
        data_path=args.data_dir + '/data/train',
        label_path=args.data_dir + '/labels/train'
    )

    global train_size
    train_size = len(train_data)

    test_data = Deidentified(
        data_path=args.data_dir + '/data/test',
        label_path=args.data_dir + '/labels/test'
    )

    global train_loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    global test_loader
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    global wv_matrix
    wv_matrix = load_wv_matrix(args.data_dir + '/wv_matrix/wv_matrix.npy')

    global criterion
    criterion = nn.CrossEntropyLoss()

    hparams = [(2, 10),      # kernel1
               (2, 10),      # kernel2
               (2, 10),      # kernel3
               (100, 200),   # num_filters1
               (100, 200),   # num_filters2
               (100, 200)]   # num_filters3

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=15,
               verbose=True,
               random_state=0)
