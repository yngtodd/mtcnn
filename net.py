import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
                nn.Conv1d(3, 64),
                nn.ReLU()
        )
