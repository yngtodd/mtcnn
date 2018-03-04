import argparse


def parse_args():
    """
    Parse Arguments for MTCNN.

    Returns:
    -------
    * `args`: [argparse object]
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='MTCNN')
    parser.add_argument('--data_dir', type=str, default='/Users/youngtodd/data/deidentified',
                        help='Root directory for the data')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to be run [default 50]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate [default: 0.01]')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum [default: 0.5]')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--half_prec', default=False,
                        help='Whether to train with half precision [default: False]')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed [default: 1]')
    args = parser.parse_args()
    return args
