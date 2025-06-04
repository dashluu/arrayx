from ..arrayx.core import Array, Backend, Shape
import numpy as np
import torch


def compare_grads(arr_grad: Array, torch_grad: torch.Tensor, name: str):
    assert torch.allclose(arr_grad.torch(), torch_grad, atol=1e-3, rtol=0), f"Gradient mismatched for {name}"


class TestBackprop:
    @classmethod
    def setup_class(cls):
        """Run once before all tests in the class"""
        print("\nSetting up TestBackprop class...")
        # Add any setup code here
        Backend.init()

    @classmethod
    def teardown_class(cls):
        """Run once after all tests in the class"""
        print("\nTearing down TestBackprop class...")
        # Add any cleanup code here
        Backend.cleanup()

    def make_shape(self) -> Shape:
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        return shape

    def test_backprop_v1(self):
        print("backprop 1:")
        shape = self.make_shape()
        np1 = np.random.randn(*shape).astype(np.float32)
        np2 = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3 = arr1 + arr2
        arr4 = arr1 * arr2
        arr5 = arr3 + arr4
        arr6 = arr3 * arr4
        arr7 = arr5 + arr6
        arr8 = arr7.sum()
        arr8.backward()
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 + t2
        t3.retain_grad()
        t4 = t1 * t2
        t4.retain_grad()
        t5 = t3 + t4
        t5.retain_grad()
        t6 = t3 * t4
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")
        compare_grads(arr5.grad, t5.grad, "arr5")
        compare_grads(arr6.grad, t6.grad, "arr6")
        compare_grads(arr7.grad, t7.grad, "arr7")

    def test_backprop_v2(self):
        print("Testing complex unary(and one binary) operations chain:")
        shape = self.make_shape()
        nparr = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)  # Positive values for log

        # Implementation: log(exp(x) * x) / x
        arr1 = Array.from_numpy(nparr)
        arr2 = arr1.exp()
        arr3 = arr2 * arr1
        arr4 = arr3.log()
        arr5 = arr4 / arr1
        arr6 = arr5.sum()
        arr6.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(nparr).requires_grad_(True)
        t2 = t1.exp()
        t2.retain_grad()
        t3 = t2 * t1
        t3.retain_grad()
        t4 = t3.log()
        t4.retain_grad()
        t5 = t4 / t1
        t6 = t5.sum()
        t6.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")

    def test_backprop_v3(self):
        print("Testing branched operations:")
        shape = self.make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3 = arr1.log()
        arr4 = arr2.exp()
        arr5 = arr3 * arr4
        arr6 = arr1 / arr2
        arr7 = arr5 + arr6
        arr8 = arr7.sum()
        arr8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1.log()
        t3.retain_grad()
        t4 = t2.exp()
        t4.retain_grad()
        t5 = t3 * t4
        t5.retain_grad()
        t6 = t1 / t2
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")
        compare_grads(arr5.grad, t5.grad, "arr5")
        compare_grads(arr6.grad, t6.grad, "arr6")
        compare_grads(arr7.grad, t7.grad, "arr7")

    def test_backprop_v4(self):
        print("Testing nested operations:")
        shape = self.make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # arrayx implementation: log(exp(x1/x2) * recip(x1))
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3 = arr1 / arr2
        arr4 = arr3.exp()
        arr5 = arr1.recip()
        arr6 = arr4 * arr5
        arr7 = arr6.log()
        arr8 = arr7.sum()
        arr8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 / t2
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = 1.0 / t1
        t5.retain_grad()
        t6 = t4 * t5
        t6.retain_grad()
        t7 = t6.log()
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")
        compare_grads(arr5.grad, t5.grad, "arr5")
        compare_grads(arr6.grad, t6.grad, "arr6")
        compare_grads(arr7.grad, t7.grad, "arr7")

    def test_backprop_v5(self):
        print("Testing square and sqrt operations:")
        shape = self.make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # arrayx implementation: sqrt(x^2) + x^2/sqrt(x)
        arr1 = Array.from_numpy(np1)
        arr2 = arr1.sq()
        arr3 = arr2.sqrt()
        arr4 = arr1.sq()
        arr5 = arr1.sqrt()
        arr6 = arr4 / arr5
        arr7 = arr3 + arr6
        arr8 = arr7.sum()
        arr8.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = t1 * t1
        t2.retain_grad()
        t3 = t2.sqrt()
        t3.retain_grad()
        t4 = t1 * t1
        t4.retain_grad()
        t5 = t1.sqrt()
        t5.retain_grad()
        t6 = t4 / t5
        t6.retain_grad()
        t7 = t3 + t6
        t7.retain_grad()
        t8 = t7.sum()
        t8.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")
        compare_grads(arr5.grad, t5.grad, "arr5")
        compare_grads(arr6.grad, t6.grad, "arr6")
        compare_grads(arr7.grad, t7.grad, "arr7")

    def test_backprop_twice(self):
        print("Testing double backpropagation with complex operations:")
        shape = self.make_shape()
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # arrayx implementation
        # f(x1, x2) = log(sqrt(x1^2) * exp(x2/x1)) + (x1 * sqrt(x2))^2
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3 = arr1.sq()
        arr4 = arr3.sqrt()
        arr5 = arr2 / arr1
        arr6 = arr5.exp()
        arr7 = arr4 * arr6
        arr8 = arr7.log()
        arr9 = arr2.sqrt()
        arr10 = arr1 * arr9
        arr11 = arr10.sq()
        arr12 = arr8 + arr11
        arr13 = arr12.sum()
        # First backward pass
        arr13.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 * t1
        t3.retain_grad()
        t4 = t3.sqrt()
        t4.retain_grad()
        t5 = t2 / t1
        t5.retain_grad()
        t6 = t5.exp()
        t6.retain_grad()
        t7 = t4 * t6
        t7.retain_grad()
        t8 = t7.log()
        t8.retain_grad()
        t9 = t2.sqrt()
        t9.retain_grad()
        t10 = t1 * t9
        t10.retain_grad()
        t11 = t10 * t10
        t11.retain_grad()
        t12 = t8 + t11
        t12.retain_grad()
        t13 = t12.sum()
        t13.backward(retain_graph=True)

        # Compare first backward pass gradients
        print("\nChecking first backward pass:")
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")
        compare_grads(arr5.grad, t5.grad, "arr5")
        compare_grads(arr6.grad, t6.grad, "arr6")
        compare_grads(arr7.grad, t7.grad, "arr7")
        compare_grads(arr8.grad, t8.grad, "arr8")
        compare_grads(arr9.grad, t9.grad, "arr9")
        compare_grads(arr10.grad, t10.grad, "arr10")
        compare_grads(arr11.grad, t11.grad, "arr11")
        compare_grads(arr12.grad, t12.grad, "arr12")

        # Modify the result slightly and run backward again
        # TODO: uncomment this after implementing backprop for broadcasting
        # result = result * 2.0
        # result.forward()
        arr13.backward()

        # PyTorch second pass
        # TODO: uncomment this after implementing backprop for broadcasting
        # torch_result = torch_result * 2.0
        # Clear all gradients from PyTorch computation graph
        t1.grad = None
        t2.grad = None
        t3.grad = None
        t4.grad = None
        t5.grad = None
        t6.grad = None
        t7.grad = None
        t8.grad = None
        t9.grad = None
        t10.grad = None
        t11.grad = None
        t12.grad = None
        t13.backward()

        # Compare second backward pass gradients
        print("\nChecking second backward pass:")
        compare_grads(arr1.grad, t1.grad, "arr1 (2nd pass)")
        compare_grads(arr2.grad, t2.grad, "arr2 (2nd pass)")
        compare_grads(arr3.grad, t3.grad, "arr3 (2nd pass)")
        compare_grads(arr4.grad, t4.grad, "arr4 (2nd pass)")
        compare_grads(arr5.grad, t5.grad, "arr5 (2nd pass)")
        compare_grads(arr6.grad, t6.grad, "arr6 (2nd pass)")
        compare_grads(arr7.grad, t7.grad, "arr7 (2nd pass)")
        compare_grads(arr8.grad, t8.grad, "arr8 (2nd pass)")
        compare_grads(arr9.grad, t9.grad, "arr9 (2nd pass)")
        compare_grads(arr10.grad, t10.grad, "arr10 (2nd pass)")
        compare_grads(arr11.grad, t11.grad, "arr11 (2nd pass)")
        compare_grads(arr12.grad, t12.grad, "arr12 (2nd pass)")

    def test_permute_binary_backprop(self):
        """Test backprop through permute and binary op"""
        # Forward: (2,3,4) -> (4,2,3) * (4,2,3)
        x = torch.randn(2, 3, 4, dtype=torch.float32)
        y = torch.randn(4, 2, 3, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = Array.from_numpy(y.numpy())
        arr3 = arr1.permute([2, 0, 1]) * arr2
        arr4 = arr3.sum()
        arr4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1.permute(2, 0, 1) * t2
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")

    def test_backprop_v6(self):
        """Test backprop through complex chain of operations"""
        print("\nTesting complex chain backprop:")
        # Forward: (2,3,4,5) -> permute -> reshape -> exp
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = arr1.permute([0, 2, 1, 3]).reshape([8, 3, 5])  # (2,4,3,5)  # (8,3,5)
        arr3 = arr2.exp()
        arr4 = arr3.sum()
        arr4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1.permute(0, 2, 1, 3).reshape(8, 3, 5)  # (2,4,3,5)  # (8,3,5)
        t2.retain_grad()
        t3 = t2.exp()
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")

    def test_backprop_v7(self):
        """Test backprop through complex chain of operations"""
        print("\nTesting complex chain backprop:")
        # Forward: (2,3,4,5) -> permute -> reshape -> matmul -> exp
        x = torch.randn(2, 3, 4, 5, dtype=torch.float32)
        # TODO: can try doing matmul with broadcast
        y = torch.randn(8, 5, 2, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = Array.from_numpy(y.numpy())
        arr3 = arr1.permute([0, 2, 1, 3]).reshape([8, 3, 5]) @ arr2  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
        arr4 = arr3.exp()
        arr5 = arr4.sum()
        arr5.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1.permute(0, 2, 1, 3).reshape(8, 3, 5) @ t2  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = t4.sum()
        t5.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")

    def test_slice_basic_backprop(self):
        """Test basic slicing backpropagation"""
        print("\nTesting basic slice backprop:")
        x = torch.randn(4, 6, 8, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = arr1[1:3, ::2, ::1]  # Basic slicing
        arr3 = arr2.sum()
        arr3.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[1:3, ::2, ::1]
        t2.retain_grad()
        t3 = t2.sum()
        t3.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")

    def test_slice_with_unary_backprop(self):
        """Test slicing combined with unary operations"""
        print("\nTesting slice with unary ops backprop:")
        x = torch.randn(3, 4, 5, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = arr1[::2, 1:3]  # Slice first
        arr3 = arr2.exp()  # Then unary op
        arr4 = arr3.sum()
        arr4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[::2, 1:3]
        t2.retain_grad()
        t3 = t2.exp()
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")

    def test_slice_with_binary_backprop(self):
        """Test slicing combined with binary operations"""
        print("\nTesting slice with binary ops backprop:")

        x = torch.randn(4, 6, 8, dtype=torch.float32)
        y = torch.randn(2, 6, 8, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = Array.from_numpy(y.numpy())
        arr3 = arr1[::2] * arr2  # Slice and multiply
        arr4 = arr3.sum()
        arr4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1[::2] * t2
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")

    def test_slice_chain_backprop(self):
        """Test chain of slice operations"""
        print("\nTesting slice chain backprop:")
        x = torch.randn(5, 6, 7, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = arr1[1:4, ::2]  # First slice
        arr3 = arr2[:, 1::2]  # Second slice
        arr4 = arr3.sum()
        arr4.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = t1[1:4, ::2]
        t2.retain_grad()
        t3 = t2[:, 1::2]
        t3.retain_grad()
        t4 = t3.sum()
        t4.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")

    def test_slice_complex_chain_backprop(self):
        """Test complex chain with slicing, unary and binary operations"""
        print("\nTesting complex slice chain backprop:")

        x = torch.randn(4, 5, 6, dtype=torch.float32)
        y = torch.randn(2, 5, 3, dtype=torch.float32)

        # arrayx implementation
        arr1 = Array.from_numpy(x.numpy())
        arr2 = Array.from_numpy(y.numpy())
        arr3 = arr1[::2, :, ::2]  # Initial slice
        arr4 = arr3.exp()  # Unary op
        arr5 = arr4 * arr2  # Binary op
        arr6 = arr5[:, 1:4]  # Another slice
        arr7 = arr6.sum()
        arr7.backward()

        # PyTorch implementation
        t1 = x.requires_grad_(True)
        t2 = y.requires_grad_(True)
        t3 = t1[::2, :, ::2]
        t3.retain_grad()
        t4 = t3.exp()
        t4.retain_grad()
        t5 = t4 * t2
        t5.retain_grad()
        t6 = t5[:, 1:4]
        t6.retain_grad()
        t7 = t6.sum()
        t7.backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")

    def test_linear_backprop(self):
        """Test backprop through matmul"""
        print("\nTesting matmul backprop:")
        np1 = np.random.randn(64, 784).astype(np.float32)
        np2 = np.random.randn(10, 784).astype(np.float32)
        np3 = np.random.randn(10).astype(np.float32)
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3 = Array.from_numpy(np3)
        arr4 = arr1 @ arr2.transpose(-2, -1) + arr3
        arr5 = arr4.sum()
        arr5.backward()
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = torch.from_numpy(np3).requires_grad_(True)
        t4 = t1 @ t2.T + t3
        t4.retain_grad()
        t5 = t4.sum()
        t5.backward()
        assert torch.allclose(arr4.torch(), t4, atol=1e-3, rtol=0)
        compare_grads(arr1.grad, t1.grad, "arr1")
        compare_grads(arr2.grad, t2.grad, "arr2")
        compare_grads(arr3.grad, t3.grad, "arr3")
