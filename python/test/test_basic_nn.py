from __future__ import annotations
from arrayx.core import Array, Backend
import nn
import numpy as np
import torch


class TestBasicNN:
    @classmethod
    def setup_class(cls):
        """Run once before all tests in the class"""
        print("\nSetting up TestBasicNN class...")
        # Add any setup code here
        Backend.init()

    @classmethod
    def teardown_class(cls):
        """Run once after all tests in the class"""
        print("\nTearing down TestBasicNN class...")
        # Add any cleanup code here
        Backend.cleanup()

    def test_linear(self):
        x = np.random.randn(64, 784).astype(np.float32)
        arr1 = Array.from_numpy(x)
        t1 = torch.from_numpy(x)
        linear = nn.Linear(784, 10)
        w = linear.w
        b = linear.b
        t2: torch.Tensor = w.torch()
        t2.requires_grad_(True)
        t2.retain_grad()
        t3: torch.Tensor = b.torch()
        t3.requires_grad_(True)
        t3.retain_grad()
        arr2 = linear(arr1).sum()
        t4 = (t1 @ t2.T + t3).sum()
        arr2.backward()
        t4.backward()
        assert torch.allclose(arr2.torch(), t4, atol=1e-3, rtol=0)

    def test_relu(self):
        x = np.random.randn(64, 10).astype(np.float32)
        arr1 = Array.from_numpy(x)
        t1 = torch.from_numpy(x)
        t1.requires_grad_(True)
        arr2 = nn.relu(arr1)
        arr3 = arr2.sum()
        t2 = torch.relu(t1)
        t3 = t2.sum()
        arr3.backward()
        t3.backward()
        assert torch.allclose(arr2.torch(), t2, atol=1e-3, rtol=0)
        assert torch.allclose(arr1.grad.torch(), t1.grad, atol=1e-3, rtol=0)

    def test_onehot(self):
        x = np.random.randint(0, 10, (64,), dtype=np.int32)
        arr1 = Array.from_numpy(x)
        t1 = torch.from_numpy(x).type(torch.int64)
        onehot1 = nn.onehot(arr1)
        onehot2 = torch.nn.functional.one_hot(t1, num_classes=10).type(torch.int32)
        assert torch.allclose(onehot1.torch(), onehot2, atol=1e-3, rtol=0)

    def test_cross_entropy(self):
        x = np.random.randn(64, 10).astype(np.float32)
        y = np.random.randint(0, 10, (64,), dtype=np.int32)
        arr1 = Array.from_numpy(x)
        t1 = torch.from_numpy(x)
        t1.requires_grad_(True)
        arr2 = Array.from_numpy(y)
        t2 = torch.from_numpy(y).type(torch.int64)
        loss1 = nn.cross_entropy_loss(arr1, arr2)
        loss2 = torch.nn.CrossEntropyLoss()(t1, t2)
        assert torch.allclose(loss1.torch(), loss2, atol=1e-3, rtol=0)
