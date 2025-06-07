from arrayx.core import Array, f32, b8
from arrayx.nn import cross_entropy_loss
from arrayx.optim import GradientDescent
from mnist import MnistModel
import ax
import numpy as np
import torch


# with ax.context():
#     x = torch.randn(2, 3, dtype=torch.float32)
#     arr1 = Array.from_numpy(x.numpy())
#     arr2 = arr1.mean()
#     arr3 = (arr1 > 0).astype(f32)
#     print(arr1)
#     print(arr3)
#     arr4 = arr2.astype(b8)
#     print(arr2.item())
#     print(arr4.item())


def test_mnist_logits():
    npx = np.random.randn(64, 784).astype(np.float32)
    npy = np.random.randint(0, 10, (64,), dtype=np.int32)
    x = Array.from_numpy(npx)
    y = Array.from_numpy(npy)
    model = MnistModel()
    logits = model.jit(x)
    loss = cross_entropy_loss(logits, y)
    optimizer = GradientDescent(model.parameters(), lr=1e-3)
    loss.backward()
    optimizer.step()
    print(loss.item())


def mnist():
    with ax.context():
        test_mnist_logits()


mnist()
