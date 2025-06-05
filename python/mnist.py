from arrayx.core import Array, f32, b8
from arrayx.optim import GradientDescent
import nn


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.linear1(input)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


def train_mnist():
    model = MnistModel()
    loss_fn = None
