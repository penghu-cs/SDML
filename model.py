import torch.nn as nn
import torch.nn.functional as F
import torch
import utils_PyTorch as utils
from torch.autograd import Variable
import math

class Dense_Net(nn.Module):
    """Generator for transfering from svhn to mnist"""

    def __init__(self, input_dim=28*28, out_dim=20):
        super(Dense_Net, self).__init__()
        mid_num = 1024
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, out_dim)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        norm_x = torch.norm(out3, dim=1, keepdim=True)
        out3 = out3 / norm_x
        return [out1, out2, out3]

