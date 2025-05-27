import torch
import numpy as np
from python.arrayx import Array, Backend


class TestReduce:
    @classmethod
    def setup_class(cls):
        """Run once before all tests in the class"""
        print("\nSetting up TestInitializers class...")
        # Add any setup code here
        Backend.init()

    @classmethod
    def teardown_class(cls):
        """Run once after all tests in the class"""
        print("\nTearing down TestInitializers class...")
        # Add any cleanup code here
        Backend.cleanup()

    @staticmethod
    def check_err(a: np.ndarray, b: np.ndarray, rtol=5e-3, atol=1e-6):
        result = np.allclose(a, b, rtol=rtol, atol=atol)
        if not result:
            # print(a.shape, b.shape)
            # Find indices where values aren't close
            close_mask = np.isclose(a, b, rtol=rtol, atol=atol)
            if a.ndim > 1:
                entry = np.where(~close_mask)
                print(a[entry], b[entry])
            else:
                entry = np.where(~close_mask)[0]
                print(a[entry], b[entry])
        assert result

    def reduce_basic(self, op1, op2):
        # Create test data
        test_cases = [[2, 3, 4], [3, 4, 5], [31, 67, 18, 17], [1027, 64, 32], [5674, 3289], [1, 1]]

        for shape in test_cases:
            print(shape)
            x = torch.randn(shape, dtype=torch.float32)
            arr1 = Array.from_numpy(x.numpy())
            arr2: Array = op1(arr1)
            arr2.eval()
            expected: torch.Tensor = op2(x)
            self.check_err(arr2.numpy(), expected.numpy())

    def reduce_2d(self, op1, op2):
        # Test cases with different dimensions
        shapes = [(2, 3), (19, 29), (37, 32), (47, 7), (57, 3), (297, 101), (5674, 3289), (256, 1), (1, 1), (1, 997)]

        for shape in shapes:
            for dim in range(2):
                x = torch.randn(*shape, dtype=torch.float32)
                arr1 = Array.from_numpy(x.numpy())
                arr2: Array = op1(arr1, [dim])
                arr2.eval()
                expected: torch.Tensor = op2(x, dim)
                self.check_err(arr2.numpy(), expected.numpy())

    def reduce_multidim(self, op1, op2):
        # Test cases with different shapes
        shapes = [
            (2, 3, 6, 4),
            (19, 29, 5, 11, 7),
            (37, 32, 18, 9),
            (47, 7, 7, 1, 3),
            (57, 3, 12),
            (297, 101, 5, 6, 3),
            (5674, 3289, 1),
        ]
        dims = [[1, 2], [0, 1, 3], [2, 3], [1, 3, 4], [0, 1, 2], [0, 1, 4], [1]]

        for shape, dim in zip(shapes, dims):
            x = torch.randn(*shape, dtype=torch.float32)
            arr1 = Array.from_numpy(x.numpy())
            arr2: Array = op1(arr1, dim)
            arr2.eval()
            expected: torch.Tensor = op2(x, dim)
            self.check_err(arr2.numpy(), expected.numpy())

    def arg_reduce_dim_in_multidim_arr(self, op1, op2):
        # Test cases with different shapes
        shapes = [
            (2, 3, 6, 4),
            (19, 29, 5, 11, 7),
            (37, 32, 18, 9),
            (47, 7, 7, 1, 3),
            (57, 3, 12),
            (297, 101, 5, 6, 3),
            (5674, 3289, 1),
        ]
        dims = [[3], [1], [2], [4], [0], [1], [0]]

        for shape, dim in zip(shapes, dims):
            x = torch.randn(*shape, dtype=torch.float32)
            arr1 = Array.from_numpy(x.numpy())
            arr2: Array = op1(arr1, dim)
            arr2.eval()
            expected: torch.Tensor = op2(x, dim=dim[0])
            self.check_err(arr2.numpy(), expected.numpy())

    def test_reduce_sum_basic(self):
        """Test basic sum reduction without specified dimensions"""
        print("\nTesting basic sum reduction:")
        self.reduce_basic(lambda x: x.sum(), lambda x: x.sum())

    def test_reduce_sum_2d(self):
        """Test sum reduction along the row and column dimension"""
        print("\nTesting sum reduction along the row and column dimension:")
        self.reduce_2d(lambda x, dim: x.sum(dim), lambda x, dim: x.sum(dim=dim).unsqueeze(dim=-1))

    def test_reduce_sum_multidim(self):
        """Test sum reduction along multiple dimensions"""
        print("\nTesting sum reduction along multiple dimensions:")
        self.reduce_multidim(lambda x, dim: x.sum(dim), lambda x, dim: x.sum(dim=dim).unsqueeze(dim=-1))

    def test_reduce_max_basic(self):
        """Test basic max reduction without specified dimensions"""
        print("\nTesting basic max reduction:")
        self.reduce_basic(lambda x: x.max(), lambda x: x.max())

    def test_reduce_max_2d(self):
        """Test max reduction along the column dimension"""
        print("\nTesting max reduction along the column dimension:")
        self.reduce_2d(lambda x, dim: x.max(dim), lambda x, dim: x.amax(dim=dim).unsqueeze(dim=-1))

    def test_reduce_max_multidim(self):
        """Test max reduction along multiple dimensions"""
        print("\nTesting max reduction along multiple dimensions:")
        self.reduce_multidim(lambda x, dim: x.max(dim), lambda x, dim: x.amax(dim=dim).unsqueeze(dim=-1))

    def test_reduce_min_basic(self):
        """Test basic min reduction without specified dimensions"""
        print("\nTesting basic min reduction:")
        self.reduce_basic(lambda x: x.min(), lambda x: x.min())

    def test_reduce_min_2d(self):
        """Test min reduction along the column dimension"""
        print("\nTesting min reduction along the column dimension:")
        self.reduce_2d(lambda x, dim: x.min(dim), lambda x, dim: x.amin(dim=dim).unsqueeze(dim=-1))

    def test_reduce_min_multidim(self):
        """Test min reduction along multiple dimensions"""
        print("\nTesting min reduction along multiple dimensions:")
        self.reduce_multidim(lambda x, dim: x.min(dim), lambda x, dim: x.amin(dim=dim).unsqueeze(dim=-1))

    def test_reduce_argmax_basic(self):
        """Test basic argmax reduction without specified dimensions"""
        print("\nTesting basic argmax reduction:")
        self.reduce_basic(lambda x: x.argmax(), lambda x: x.argmax())

    def test_reduce_argmax_2d(self):
        """Test argmax reduction along the column dimension"""
        print("\nTesting argmax reduction along the column dimension:")
        self.reduce_2d(lambda x, dim: x.argmax(dim), lambda x, dim: x.argmax(dim=dim).unsqueeze(dim=-1))

    def test_reduce_argmax_multidim(self):
        """Test argmax reduction along multiple dimensions"""
        print("\nTesting argmax reduction along multiple dimensions:")
        self.arg_reduce_dim_in_multidim_arr(
            lambda x, dim: x.argmax(dim), lambda x, dim: x.argmax(dim=dim).unsqueeze(dim=-1)
        )

    def test_reduce_argmin_basic(self):
        """Test basic argmin reduction without specified dimensions"""
        print("\nTesting basic argmin reduction:")
        self.reduce_basic(lambda x: x.argmin(), lambda x: x.argmin())

    def test_reduce_argmin_2d(self):
        """Test argmin reduction along the column dimension"""
        print("\nTesting argmin reduction along the column dimension:")
        self.reduce_2d(lambda x, dim: x.argmin(dim), lambda x, dim: x.argmin(dim=dim).unsqueeze(dim=-1))

    def test_reduce_argmin_multidim(self):
        """Test argmin reduction along multiple dimensions"""
        print("\nTesting argmin reduction along multiple dimensions:")
        self.arg_reduce_dim_in_multidim_arr(
            lambda x, dim: x.argmin(dim), lambda x, dim: x.argmin(dim=dim).unsqueeze(dim=-1)
        )
