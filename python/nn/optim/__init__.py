from collections.abc import Sequence
from ...arrayx import Array


# Vanilla Gradient Descent
class VGD:
    def __init__(self, params: Sequence[Array], lr=1e-3):
        self.__params: list[Array] = []
        self.__lr = lr
        self.__init_graph = False
        for param in params:
            self.__params.append(param.detach())

    def step(self):
        if not self.__init_graph:
            for i in range(len(self.__params)):
                self.__params[i] -= self.__params[i].grad * self.__lr
            self.__init_graph = True
        for param in self.__params:
            param.eval()
