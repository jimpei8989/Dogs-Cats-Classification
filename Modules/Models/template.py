from torch import nn
from torchsummary import summary

class CNN_Template(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = None

    def forward(self, x):
        return self.network(x)

    def summary(self):
        summary(self.network, (3, 64, 64))

