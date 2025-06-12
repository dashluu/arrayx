from arrayx.core import Array
import numpy as np


class TestTransform:
    def test_slice_v1(self):
        print("slice 1:")
        shape = [np.random.randint(1, 50) for _ in range(3)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1[::, ::, ::]
        npslice1 = arr2.numpy()
        npslice2 = nparr[::, ::, ::]
        assert np.allclose(npslice1, npslice2, atol=1e-3, rtol=0)

    def test_slice_v2(self):
        print("slice 2:")
        shape = [np.random.randint(4, 50) for _ in range(4)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1[1::4, :3:2, 2::3]
        npslice1 = arr2.numpy()
        npslice2 = nparr[1::4, :3:2, 2::3]
        assert np.allclose(npslice1, npslice2, atol=1e-3, rtol=0)

    def test_slice_v3(self):
        print("slice 3:")
        shape = [np.random.randint(4, 50) for _ in range(4)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1[1::, ::2, 3:0:-2]
        npslice1 = arr2.numpy()
        npslice2 = nparr[1::, ::2, 3:0:-2]
        assert np.allclose(npslice1, npslice2, atol=1e-3, rtol=0)

    def test_slice_v4(self):
        print("slice 4:")
        shape = [np.random.randint(10, 50) for _ in range(4)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1[1:0:-4, 9:3:-2, 2::3]
        npslice1 = arr2.numpy()
        npslice2 = nparr[1:0:-4, 9:3:-2, 2::3]
        assert np.allclose(npslice1, npslice2, atol=1e-3, rtol=0)

    def test_transpose_start(self):
        print("transpose at the start:")
        shape = [np.random.randint(3, 10) for _ in range(4)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1.transpose(0, 2)
        nptranspose1 = arr2.numpy()
        order = list(range(len(shape)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[0 : 2 + 1] = order[0 : 2 + 1][::-1]  # [2,1,0,3]
        nptranspose2 = np.transpose(nparr, order)
        assert np.allclose(nptranspose1, nptranspose2, atol=1e-3, rtol=0)

    def test_transpose_mid(self):
        print("transpose in the middle:")
        shape = [np.random.randint(3, 10) for _ in range(6)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1.transpose(1, -2)
        nptranspose1 = arr2.numpy()
        order = list(range(len(shape)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[1:-1] = order[1:-1][::-1]  # [0,3,2,1]
        nptranspose2 = np.transpose(nparr, order)
        assert np.allclose(nptranspose1, nptranspose2, atol=1e-3, rtol=0)

    def test_transpose_end(self):
        print("transpose at the end:")
        shape = [np.random.randint(3, 10) for _ in range(5)]
        nparr = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1.transpose(-3, -1)
        nptranspose1 = arr2.numpy()
        order = list(range(len(shape)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[-3:] = order[-3:][::-1]  # [0,3,2,1]
        nptranspose2 = np.transpose(nparr, order)
        assert np.allclose(nptranspose1, nptranspose2, atol=1e-3, rtol=0)

    def test_permute(self):
        print("\nTesting permute operations:")

        # Test cases: [(shape, permutation)]
        test_cases = [
            # Basic permutations
            ([2, 3, 4], [2, 0, 1]),  # 3D rotation
            ([2, 3, 4, 5], [3, 2, 1, 0]),  # Complete reverse
            ([2, 3, 4, 5], [0, 2, 1, 3]),  # Middle swap
            # Edge cases
            ([1, 2, 3], [2, 1, 0]),  # With dimension size 1
            ([5, 1, 1, 4], [0, 2, 1, 3]),  # Multiple size-1 dimensions
            ([2, 3], [1, 0]),  # 2D transpose
        ]

        for shape, permutation in test_cases:
            print(f"\nTesting shape {shape} with permutation {permutation}")

            # Create test data
            nparr = np.random.randn(*shape).astype(np.float32)
            arr1 = Array.from_numpy(nparr)
            arr2 = arr1.permute(permutation)
            nppermute1 = arr2.numpy()
            nppermute2 = np.transpose(nparr, permutation)

            # Verify results
            assert np.allclose(
                nppermute1, nppermute2, atol=1e-3, rtol=0
            ), f"Permute failed for shape {shape} with perm {permutation}"
            assert (
                nppermute1.shape == nppermute2.shape
            ), f"Shape mismatch: got {nppermute1.shape}, expected {nppermute2.shape}"

    def test_flatten(self):
        print("\nTesting flatten operations:")

        # Test cases: [(shape, start_dim, end_dim, expected_shape)]
        test_cases = [
            # Basic flattening
            ([2, 3, 4], 0, -1, [24]),  # Flatten all
            ([2, 3, 4, 5], 1, 2, [2, 12, 5]),  # Middle flatten
            ([2, 3, 4, 5], 0, 1, [6, 4, 5]),  # Start flatten
            ([2, 3, 4, 5], -2, -1, [2, 3, 20]),  # End flatten
            # Edge cases
            ([1, 2, 3, 4], 1, 3, [1, 24]),  # With leading 1
            ([2, 1, 3, 1], 1, 2, [2, 3, 1]),  # With middle 1s
            ([5], 0, 0, [5]),  # Single dimension
        ]

        for shape, start, end, expected in test_cases:
            print(f"\nTesting shape {shape} flatten({start},{end})")

            # Create test data
            nparr = np.random.randn(*shape).astype(np.float32)
            arr1 = Array.from_numpy(nparr)
            arr2 = arr1.flatten(start, end)
            npflatten1 = arr2.numpy()
            npflatten2 = nparr.reshape(expected)

            # Verify results
            assert np.allclose(
                npflatten1, npflatten2, atol=1e-3, rtol=0
            ), f"Flatten failed for shape {shape} with dims {start},{end}"
            assert npflatten1.shape == tuple(expected), f"Shape mismatch: got {npflatten1.shape}, expected {expected}"
