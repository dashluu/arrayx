from __future__ import annotations
from arrayx.core import Array
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
    def binary_no_broadcast(self, name: str, op1, op2, gen=randn):
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = gen(shape)
        np2 = gen(shape)
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3: Array = op1(arr1, arr2)
        np3: np.ndarray = arr3.numpy()
        np4: np.ndarray = op2(np1, np2)
        assert tuple(arr3.view) == np4.shape
        assert np.allclose(np3, np4, atol=1e-3, rtol=0)

    def binary_with_broadcast(self, name: str, op1, op2, gen=randn):
        print(f"{name} with broadcast:")
        # Test cases with different broadcasting scenarios
        test_cases = [
            # [shape1, shape2, result_shape]
            ([2, 1, 4], [3, 4], [2, 3, 4]),  # Left broadcast
            ([1, 5], [2, 1, 5], [2, 1, 5]),  # Right broadcast
            ([3, 1, 1], [1, 4, 5], [3, 4, 5]),  # Both broadcast
            ([1], [2, 3, 4], [2, 3, 4]),  # Scalar to array
            ([2, 3, 4], [1], [2, 3, 4]),  # Array to scalar
            ([3, 1, 19, 1, 1], [1, 47, 19, 63, 1], [3, 47, 19, 63, 1]),
            ([1, 2], [1, 47, 19, 63, 1], [1, 47, 19, 63, 2]),
        ]
        for shape1, shape2, expected_shape in test_cases:
            print(f"\nTesting shapes: {shape1}, {shape2} -> {expected_shape}")
            np1 = gen(shape1)
            np2 = gen(shape2)
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)
            arr3: Array = op1(arr1, arr2)
            np3: np.ndarray = arr3.numpy()
            np4: np.ndarray = op2(np1, np2)
            assert tuple(arr3.view) == np4.shape
            assert np.allclose(np3, np4, atol=1e-3, rtol=0)

    def binary_inplace(self, name: str, op1, op2, gen=randn):
        print(f"{name} inplace:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]

        # Generate inputs
        np1: np.ndarray = gen(shape)
        np2: np.ndarray = gen(shape)
        np1_copy = np1.copy()  # Keep copy for numpy comparison

        # Create arrays
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)

        # Apply inplace operation
        arr1: Array = op1(arr1, arr2)  # arr1 += arr2, etc.
        arr1: Array = op1(arr1, arr2)  # Second time to make sure it's updated.

        # Compare with NumPy
        np1_copy: np.ndarray = op2(np1_copy, np2)  # np1_copy += np2, etc.
        np1_copy: np.ndarray = op2(np1_copy, np2)  # Second time
        assert tuple(arr1.view) == np1_copy.shape
        assert np.allclose(arr1.numpy(), np1_copy, atol=1e-3, rtol=0)

    def binary_inplace_broadcast(self, name: str, op1, op2, gen=randn):
        print(f"{name} inplace broadcast:")

        test_cases = [
            # [lhs_shape, rhs_shape] -> result shape will be lhs_shape
            ([2, 3, 4], [4]),  # Broadcast scalar to 3D
            ([3, 4, 5], [1, 5]),  # Broadcast from 2D to 3D
            ([2, 4, 6], [4, 1]),  # Broadcast with ones
            ([5, 5, 5], [1, 5, 1]),  # Broadcast with ones in multiple dims
            ([4, 3, 2], [3, 1]),  # Partial broadcast with ones
            ([3, 47, 19, 63, 1], [1, 1, 19, 63, 1]),
            ([1, 47, 19, 63, 2], [1, 2]),
        ]

        for lhs_shape, rhs_shape in test_cases:
            print(f"\nTesting: {lhs_shape} @= {rhs_shape}")

            # Generate inputs
            np1: np.ndarray = gen(lhs_shape)
            np2: np.ndarray = gen(rhs_shape)
            np1_copy = np1.copy()

            # Create arrays
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)

            # Apply inplace operation
            arr1: Array = op1(arr1, arr2)
            # Compare with NumPy
            np1_copy: np.ndarray = op2(np1_copy, np2)
            assert tuple(arr1.view) == np1_copy.shape
            assert np.allclose(arr1.numpy(), np1_copy, atol=1e-3, rtol=0)

    def test_add(self):
        self.binary_no_broadcast("add", operator.add, operator.add)

    def test_sub(self):
        self.binary_no_broadcast("sub", operator.sub, operator.sub)

    def test_mul(self):
        self.binary_no_broadcast("mul", operator.mul, operator.mul)

    def test_div(self):
        self.binary_no_broadcast("div", operator.truediv, operator.truediv)

    def test_minimum(self):
        self.binary_no_broadcast("minimum", lambda x, y: x.minimum(y), lambda x, y: np.minimum(x, y))

    def test_maximum(self):
        self.binary_no_broadcast("maximum", lambda x, y: x.maximum(y), lambda x, y: np.maximum(x, y))

    def test_add_broadcast(self):
        self.binary_with_broadcast("add", operator.add, operator.add)

    def test_sub_broadcast(self):
        self.binary_with_broadcast("sub", operator.sub, operator.sub)

    def test_mul_broadcast(self):
        self.binary_with_broadcast("mul", operator.mul, operator.mul)

    def test_div_broadcast(self):
        self.binary_with_broadcast("div", operator.truediv, operator.truediv)

    def test_add_inplace(self):
        self.binary_inplace("add", operator.iadd, operator.iadd)

    def test_sub_inplace(self):
        self.binary_inplace("sub", operator.isub, operator.isub)

    def test_mul_inplace(self):
        self.binary_inplace("mul", operator.imul, operator.imul)

    def test_div_inplace(self):
        self.binary_inplace("div", operator.itruediv, operator.itruediv)

    def test_add_inplace_broadcast(self):
        self.binary_inplace_broadcast("add", operator.iadd, operator.iadd)

    def test_sub_inplace_broadcast(self):
        self.binary_inplace_broadcast("sub", operator.isub, operator.isub)

    def test_mul_inplace_broadcast(self):
        self.binary_inplace_broadcast("mul", operator.imul, operator.imul)

    def test_div_inplace_broadcast(self):
        self.binary_inplace_broadcast("div", operator.itruediv, operator.itruediv)
