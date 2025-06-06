from arrayx.core import Array, f32, b8
from arrayx.optim import GradientDescent
import nn
import numpy as np
import ax


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x: Array):
        x = self.linear1(x)
        # x = nn.relu(x)
        # x = self.linear2(x)
        return x

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()


def test_mnist_logits():
    npx = np.random.randn(64, 784).astype(np.float32)
    x = Array.from_numpy(npx)
    model = MnistModel()
    logits = model(x)
    print(logits)


def mnist():
    with ax.context():
        test_mnist_logits()


mnist()
