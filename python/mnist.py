from arrayx.core import Array
from arrayx.nn import relu
import nn


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: Array):
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x
