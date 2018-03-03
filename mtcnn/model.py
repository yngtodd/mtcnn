import torch
import torch.nn as nn
import torch.nn.functional as F


class MTCNN(nn.Module):
    def __init__(self, wv_matrix, kernel1, kernel2, kernel3, num_filters1,
                 num_filters2, num_filters3, dropout1, dropout2, dropout3,
                 max_sent_len, word_dim, vocab_size,
                 subsite_size, laterality_size, behavior_size, grade_size,
                 model_type='static'):
        super(MTCNN, self).__init__()
        """
        Multi-task CNN model for document classification.

        Parameters:
        ----------
        * `wv_matrix` []
            Word vector matrix

        * `kernel*`: [int]
            Kernel filter size at convolution *. 
        
        * `num_filters*` [int]
            Number of convolutional filters at convolution *.

        * `dropout*`: [float]
            Probability of elements being zeroed at convolution *.

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

        * `model_type`: [str]
            Alternative type of model being used.
            -Options:
                "static": 
                "multichannel":
        """
        self.wv_matrix = wv_matrix
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.num_filters3 = num_filters3
        self.filter_sum = _sum_filters()
        self.max_sent_len = max_sent_len
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.subsite_size = subsite_size
        self.laterality_size = laterality_size
        self.behavior_size = behavior_size
        self.grade_size = grade_size

        self.embedding = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        if self.model_type == 'static':
            self.embedding.weight.requires_grad = False
        elif self.MODEL == "multichannel":
            self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
            self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            self.embedding2.weight.requires_grad = False
            self.IN_CHANNEL = 2

        self.convblock1 = nn.Sequential(
            nn.Conv1d(1, self.num_filters1, self.kernel1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.num_filters1) 
            nn.Dropout(p=self.droput1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv1d(1, self.num_filters2, self.kernel2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(sef.num_filters2)
            nn.Dropout(p=self.droput2)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv1d(1, self.num_filters3, self.kernel3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.num_filters3)
            nn.Dropout(p=self.droput3)
        )

        self.fc1 = nn.Linear(self.filter_sum, self.subsite_size)
        self.fc2 = nn.Linear(self.filter_sum, self.laterality_size)
        self.fc3 = nn.Linear(self.filter_sum, self.behavior_size)
        self.fc4 = nn.Linear(self.filter_sum, self.self.grade_size)

        def _sum_filters(self):
            """Get the total number of convolutional filters."""
            return self.num_filters1 + self.num_filters2 + self.num_filters3
        
        def forward(self, x):
            x = self.embedding().view(-1, 1, self.word_dim * self.max_sent_len)
            if self.model_type == "multichannel":
                x2 = self.embedding2(x).view(-1, 1, self.word_dim * self.max_sent_len)
                x = torch.cat((x, x2), 1)

            conv_results = []
            conv_results.append(self.convblock1(x).view(-1, self.num_filters1))
            conv_results.append(self.convblock2(x).view(-1, self.num_filters2))
            conv_results.append(self.convblock3(x).view(-1, self.num_filters3))
            x = torch.cat(conv_results, 1)

            out_subsite = self.fc1(x)
            out_laterality = self.fc2(x)
            out_behavior = self.fc3(x)
            out_grade = self.fc4(x)
            return out_subsite, out_laterality, out_behavior, out_grade
