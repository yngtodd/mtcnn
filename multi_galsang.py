import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_ZERO_SIZE = kwargs["CLASS_ZERO_SIZE"]
        self.CLASS_ONE_SIZE = kwargs["CLASS_ONE_SIZE"]
        self.CLASS_TWO_SIZE = kwargs["CLASS_TWO_SIZE"]
        self.CLASS_THREE_SIZE = kwargs["CLASS_THREE_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=0)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2
        self.conv_out = [nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM) for i in range(len(self.FILTERS))]
        '''
        #self.conv_out = []
        for i in range(len(self.FILTERS)):
            self.conv_out.append(nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM))
            #conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            #setattr(self, 'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        '''
        for i,_ in enumerate(self.FILTERS):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_'+str(i), conv)

        # Making Extensions for Multi-Task here:
        self.fc1 = nn.Linear(sum(self.FILTER_NUM), self.CLASS_ZERO_SIZE)
        self.fc2 = nn.Linear(sum(self.FILTER_NUM), self.CLASS_ONE_SIZE)
        self.fc3 = nn.Linear(sum(self.FILTER_NUM), self.CLASS_TWO_SIZE)
        self.fc4 = nn.Linear(sum(self.FILTER_NUM), self.CLASS_THREE_SIZE)
        #self.fc5 = nn.Linear(sum(self.FILTER_NUM), self.CLASS_FOUR_SIZE)

    def get_conv(self, i):
        return getattr(self, 'conv_'+str(i))

    '''
    def get_conv(self, i):
        return self.conv_out[i]
    '''
    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        out_task1 = self.fc1(x)
        out_task2 = self.fc2(x)
        out_task3 = self.fc3(x)
        out_task4 = self.fc4(x)
        return out_task1, out_task2, out_task3, out_task4 