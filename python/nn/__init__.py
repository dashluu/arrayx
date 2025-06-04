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


def relu(x: Array):
    return (x >= 0).astype(f32) * x


def onehot(x: Array, num_classes: int = -1):
    if x.dtype.type != DtypeType.INT:
        raise ValueError(f"Array {x.id} must be of type int.")
    if num_classes == -1:
        max_x = x.max()
        num_classes = max_x.item() + 1
    arange = Array.arange([num_classes], 0, 1, i32)
    return (x.unsqueeze() == arange).astype(i32)


def cross_entropy_loss(x: Array, y: Array, num_classes: int = -1):
    # x is logits, y is target
    # compute cross entropy loss
    # -sum(y * log(softmax(x)))
    # softmax(x) = exp(x) / sum(exp(x))
    # log(softmax(x)) = x - log(sum(exp(x)))
    # loss = -sum(y * (x - log(sum(exp(x)))))
    # loss = -sum(y * x) + sum(y * log(sum(exp(x))))
    # sum(y) = 1 and log(sum(exp(x))) is a scalar
    # loss = -sum(y * x) + log(sum(exp(x)))
    # logsumexp trick for numerical stability:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # x: (*, N)
    # y: (*, 1) for labels
    # max_x: (*, 1)
    # exp_x: (*, N)
    # sum_exp_x: (*, 1)
    # log_sum_exp_x: (*, 1)
    # onehot_y: (*, N)
    # onehot_y * x: (*, N)
    # loss: (*, 1)
    max_x = x.max([-1])
    exp_x = (x - max_x).exp()
    sum_exp_x = exp_x.sum([-1])
    log_sum_exp_x = sum_exp_x.log() + max_x
    onehot_y = onehot(y, num_classes).astype(f32)
    loss = -(onehot_y * x).sum([-1]) + log_sum_exp_x
    return loss.mean()
