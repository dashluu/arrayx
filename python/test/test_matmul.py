import numpy as np
from arrayx.core import Array, Backend


class TestMatmul:
    @classmethod
    def setup_class(cls):
        """Run once before all tests in the class"""
        print("\nSetting up TestMatmul class...")
        # Add any setup code here
        Backend.init()

    @classmethod
    def teardown_class(cls):
        """Run once after all tests in the class"""
        print("\nTearing down TestMatmul class...")
        # Add any cleanup code here
        Backend.cleanup()

    def test_2d_matmul(self):
        """Test matrix multiplication for 2D arrays"""
        print("matmul 2d:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([2, 3], [3, 4]),  # Basic matrix multiplication
            ([1, 4], [4, 5]),  # Single row matrix
            ([3, 2], [2, 1]),  # Result is a column matrix
            ([5, 5], [5, 5]),  # Square matrices
            ([1, 1], [1, 1]),  # 1x1 matrices
        ]

        for shape1, shape2 in test_cases:
            print(f"\nTesting shapes: {shape1} @ {shape2}")
            # Generate random matrices
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)
            arr3 = arr1 @ arr2
            np3 = np1 @ np2
            # Verify shape and values
            assert tuple(arr3.view) == np3.shape
            assert np.allclose(arr3.numpy(), np3, atol=1e-3, rtol=0)

    def test_3d_matmul(self):
        """Test matrix multiplication for 3D arrays (batched matmul)"""
        print("\nTesting 3D matrix multiplication:")

        # Test cases: [(shape1, shape2, description)]
        test_cases = [
            # Basic batch matmul
            ([4, 2, 3], [4, 3, 4], "Standard batch size"),
            ([1, 2, 3], [1, 3, 4], "Single batch"),
            ([10, 3, 3], [10, 3, 3], "Square matrices batch"),
            # Broadcasting cases
            ([1, 2, 3], [5, 3, 4], "Broadcast first dim"),
            ([5, 2, 3], [1, 3, 4], "Broadcast second dim"),
            ([7, 1, 3], [7, 3, 5], "Batch with singular dimension"),
            # Edge cases
            ([3, 1, 4], [3, 4, 1], "Result has singular dimension"),
            ([2, 5, 1], [2, 1, 3], "Inner dimension is 1"),
            ([1, 1, 1], [1, 1, 1], "All dimensions are 1"),
        ]

        for shape1, shape2, desc in test_cases:
            print(f"\nTesting {desc}:")
            print(f"Shapes: {shape1} @ {shape2}")

            # Generate random matrices
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)
            arr3 = arr1 @ arr2
            np3 = np1 @ np2
            # Verify shape and values
            assert tuple(arr3.view) == np3.shape, f"Shape mismatch: got {arr3.view()}, expected {np3.shape}"
            assert np.allclose(arr3.numpy(), np3, atol=1e-3, rtol=0), f"Value mismatch for {desc}"

    def test_multidim_matmul(self):
        """Test multi-dimensional matrix multiplication"""
        print("\nTesting multi-dimensional matrix multiplication:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([1, 5, 4, 2, 3], [5, 1, 4, 3, 4]),
            ([1, 1, 2, 3], [5, 2, 3, 4]),
            ([1, 3, 7, 3, 17], [10, 3, 1, 17, 6]),
            ([13, 4, 2, 9, 1], [1, 1, 2, 1, 8]),
        ]

        for shape1, shape2 in test_cases:
            print(f"Shapes: {shape1} @ {shape2}")

            # Generate random matrices
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)
            arr3 = arr1 @ arr2
            np3 = np1 @ np2
            # Verify shape and values
            assert tuple(arr3.view) == np3.shape, f"Shape mismatch: got {arr3.view()}, expected {np3.shape}"
            assert np.allclose(arr3.numpy(), np3, atol=1e-3, rtol=0), f"Value mismatch for {shape1} @ {shape2}"
