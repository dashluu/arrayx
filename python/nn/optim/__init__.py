from collections.abc import Sequence
from ..arrayx.core import Array


# Vanilla Gradient Descent
class VGD:
    def __init__(self, params: Sequence[Array], lr=1e-3):
        self.__params: list[Array] = []
        for p in params:
            param = p.detach()
            param -= param.grad * lr
            self.__params.append(param)

    def step(self):
        for param in self.__params:
            param.eval()
