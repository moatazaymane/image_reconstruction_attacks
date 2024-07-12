import torch
import torch.nn as nn


class Softmax_Model(torch.nn.Module):

    def __init__(self, width, out_features):
        super().__init__()
        self.map = torch.nn.Linear(width, out_features)

    def forward(self, x):
        return torch.log_softmax(self.map(x), dim=-1)
