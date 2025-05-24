from __future__ import annotations
from python.arrayx import Array, Backend
import numpy as np
import operator


def randn(shape) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32)


def nonzero_randn(shape) -> np.ndarray:
    arr = randn(shape)
    # Replace zeros with small random values
    zero_mask = arr == 0
    arr[zero_mask] = np.random.uniform(0.1, 1.0, size=np.count_nonzero(zero_mask))
    return arr


def positive_randn(shape) -> np.ndarray:
    arr = nonzero_randn(shape)
    return np.abs(arr)


class TestBinary:
    @classmethod
    def setup_class(cls):
        """Run once before all tests in the class"""
        print("\nSetting up TestBinary class...")
        # Add any setup code here
        Backend.init()

    @classmethod
    def teardown_class(cls):
        """Run once after all tests in the class"""
        print("\nTearing down TestBinary class...")
        # Add any cleanup code here
        Backend.cleanup()

    def binary_no_broadcast(self, name: str, op, gen=randn):
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = gen(shape)
        np2 = gen(shape)
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3: Array = op(arr1, arr2)
        arr3.eval()
        np3 = arr3.numpy()
        np4: np.ndarray = op(np1, np2)
        assert tuple(arr3.view) == np4.shape
        assert np.allclose(np3, np4.flatten(), atol=1e-3, rtol=0)

    def test_add(self):
        self.binary_no_broadcast("add", operator.add)
