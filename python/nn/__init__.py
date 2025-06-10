from __future__ import annotations
import numpy as np
from arrayx.core import Array, DtypeType, f32, i32
from arrayx.nn import linear, linear_with_bias
from collections.abc import Sequence
from typing import Callable


class JitKey:
    @staticmethod
    def __canonicalize(item):
        if isinstance(item, Array):
            return (tuple(item.view), item.dtype.name, item.device.name)
        return item

    def __init__(self, *args, **kwargs):
        canonical_args = []
        canonical_kwargs = []
        for arg in args:
            canonical_args.append(JitKey.__canonicalize(arg))
        for key, value in kwargs.items():
            canonical_kwargs.append((key, JitKey.__canonicalize(value)))
        canonical_kwargs.sort()
        self.__canonical_form = tuple(canonical_args + canonical_kwargs)

    def __eq__(self, other: JitKey) -> bool:
        return self.__canonical_form == other.__canonical_form

    def __hash__(self) -> int:
        return hash(self.__canonical_form)


class Jit:
    def __init__(self, callable: Callable):
        self.__cache = {}
        self.__callable = callable

    def __call__(self, *args, **kwargs) -> Array:
        key = JitKey(*args, **kwargs)
        if key in self.__cache:
            return self.__cache[key]
        else:
            result = self.__callable(*args, **kwargs)
            result.compile()
            self.__cache[key] = result
            return result


class Module:
    def parameters(self) -> list[Array]:
        params = []
        for _, value in self.__dict__.items():
            if isinstance(value, Array):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")


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
        return linear(x, self.__w) if self.__b is None else linear_with_bias(x, self.__w, self.__b)
