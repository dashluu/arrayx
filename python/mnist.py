from arrayx.core import Array
from arrayx.nn import relu
from collections.abc import Sequence
import nn


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: Sequence[Array]):
        x: Array = self.linear1(x)
        x: Array = relu(x)
        x: Array = self.linear2([x])
        return x

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()
