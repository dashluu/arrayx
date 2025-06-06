import numpy as np
from arrayx.core import Array, DtypeType, f32, i32
from arrayx.nn import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        k = np.sqrt(1 / in_features)
        # Use numpy to randomize for now
        self.__npw = np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.__w = Array.from_numpy(self.__npw)
        if bias:
            self.__npb = np.random.uniform(-k, k, (out_features)).astype(np.float32)
            self.__b = Array.from_numpy(self.__npb)
        else:
            self.__b = None

    @property
    def w(self):
        return self.__w

    @property
    def b(self):
        return self.__b

    def forward(self, x: Array):
        x = x @ self.__w.transpose(-2, -1)
        return x if self.__b is None else x + self.__b

    def parameters(self):
        return [self.__w] if self.__b is None else [self.__w, self.__b]
