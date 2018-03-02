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
    parser.add_argument('--data_dir', type=str, default='/Users/youngtodd/data/deidentified'
                        help='Root directory for the data')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to be run.')
    args = parser.parse_args()
    return args
