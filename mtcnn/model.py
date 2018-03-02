import torch
import torch.nn as nn
import torch.nn.functional as F


class MTCNN(nn.Module):
    def __init__(self, wv_matrix, max_sent_len, word_dim, vocab_size,
                 subsite_size, laterality_size, behavior_size, grade_size):
        super(MTCNN, self).__init__()
        """
        Multi-task CNN model for document classification.

        Parameters:
        ----------
        * `wv_matrix` []
            Word vector matrix

        * `max_sent_len [int]
            Maximum sentence length.

        * `word_dim` [int]
            Word dimension.

        * `vocab_size`: [int]
            Vocabulary size.

        * `subsite_size`: [int]
            Class size for subsite task.

        * `laterality_size`: [int]
            Class size for laterality task.

        * `behavior_size`: [int]
            Class size for behavior task.

        * `grade_size`: [int]
            Class size for grade task.
        """
        self.wv_matrix = wv_matrix
        self.max_sent_len = max_sent_len
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.subsite_size = subsite_size
        self.laterality_size = laterality_size
        self.behavior_size = behavior_size
        self.grade_size = grade_size:
