def print_progress(epoch, loss, batch_size, train_size):
    """
    Print the learning progress.

    Parameters:
    ----------
    * `epoch`: [int]
        Epoch number.
    
    * `loss`: [float]
        Loss value.

    * `batch_size`: [int]
        Batch size for training to keep track of progress.

    * `train_size`: [int]
        Size of the training set.
    """
    print('Epoch: {:d}, Step: [{:d}/{:d}], Loss: {:.4f}' \
          .format(epoch, batch_size, train_size, loss))

