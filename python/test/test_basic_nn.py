from __future__ import annotations
from arrayx.core import Array, Backend
from arrayx.nn import relu, onehot, cross_entropy_loss
from arrayx.optim import GradientDescent
from mnist import MnistModel
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
        arr2 = relu(arr1)
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
        onehot1 = onehot(arr1)
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
        loss1 = cross_entropy_loss(arr1, arr2)
        loss2: torch.Tensor = torch.nn.CrossEntropyLoss()(t1, t2)
        loss1.backward()
        loss2.backward()
        assert torch.allclose(loss1.torch(), loss2, atol=1e-3, rtol=0)
        assert torch.allclose(arr1.grad.torch(), t1.grad, atol=1e-3, rtol=0)

    def test_single_pass(self):
        # Input data
        x = np.random.randn(64, 784).astype(np.float32)
        y = np.random.randint(0, 10, (64,), dtype=np.int32)

        # Array implementation
        arr1 = Array.from_numpy(x)
        arr2 = Array.from_numpy(y)
        model = MnistModel()

        # PyTorch implementation
        torch_model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

        # Share weights between Array and PyTorch
        # First layer
        w1: torch.Tensor = model.linear1.w.torch()
        b1: torch.Tensor = model.linear1.b.torch()
        w1.requires_grad_(True)
        b1.requires_grad_(True)
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        # Second layer
        w2: torch.Tensor = model.linear2.w.torch()
        b2: torch.Tensor = model.linear2.b.torch()
        w2.requires_grad_(True)
        b2.requires_grad_(True)
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)

        # Forward pass
        t1 = torch.from_numpy(x)
        t2 = torch.from_numpy(y).type(torch.int64)

        logits = model(arr1)
        torch_logits = torch_model(t1)

        # Loss computation
        loss = cross_entropy_loss(logits, arr2)
        torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(torch_logits, t2)

        # Backward pass
        loss.backward()
        torch_loss.backward()

        # Compare results
        assert torch.allclose(logits.torch(), torch_logits, atol=1e-3, rtol=0)
        assert torch.allclose(loss.torch(), torch_loss, atol=1e-3, rtol=0)

        # Compare gradients
        assert torch.allclose(model.linear1.w.grad.torch(), torch_model[0].weight.grad, atol=1e-3, rtol=0)
        assert torch.allclose(model.linear1.b.grad.torch(), torch_model[0].bias.grad, atol=1e-3, rtol=0)
        assert torch.allclose(model.linear2.w.grad.torch(), torch_model[2].weight.grad, atol=1e-3, rtol=0)
        assert torch.allclose(model.linear2.b.grad.torch(), torch_model[2].bias.grad, atol=1e-3, rtol=0)

    def test_multipass_with_optimizer(self):
        # Input data
        x = np.random.randn(64, 784).astype(np.float32)
        y = np.random.randint(0, 10, (64,), dtype=np.int32)

        # Array implementation
        arr1 = Array.from_numpy(x)
        arr2 = Array.from_numpy(y)
        model = MnistModel()

        # PyTorch implementation
        torch_model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

        # Share weights between Array and PyTorch
        # First layer
        w1: torch.Tensor = model.linear1.w.torch()
        b1: torch.Tensor = model.linear1.b.torch()
        w1.requires_grad_(True)
        b1.requires_grad_(True)
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        # Second layer
        w2: torch.Tensor = model.linear2.w.torch()
        b2: torch.Tensor = model.linear2.b.torch()
        w2.requires_grad_(True)
        b2.requires_grad_(True)
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)

        # Forward pass
        t1 = torch.from_numpy(x)
        t2 = torch.from_numpy(y).type(torch.int64)
        passes = 3

        for _ in range(passes):
            logits = model(arr1)
            torch_logits = torch_model(t1)
            # Loss computation
            loss = cross_entropy_loss(logits, arr2)
            torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(torch_logits, t2)
            # Backward pass
            loss.backward()
            torch_loss.backward()
            # Setup optimizers
            optimizer = GradientDescent(model.parameters(), lr=1e-3)
            torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-3, momentum=0, weight_decay=0)
            optimizer.step()
            torch_optimizer.step()
            # Compare updated weights and biases
            assert torch.allclose(model.linear1.w.torch(), torch_model[0].weight.data, atol=1e-3, rtol=0)
            assert torch.allclose(model.linear1.b.torch(), torch_model[0].bias.data, atol=1e-3, rtol=0)
            assert torch.allclose(model.linear2.w.torch(), torch_model[2].weight.data, atol=1e-3, rtol=0)
            assert torch.allclose(model.linear2.b.torch(), torch_model[2].bias.data, atol=1e-3, rtol=0)

    def test_multipass_with_jit_and_optimizer(self):
        # Input data
        x = np.random.randn(64, 784).astype(np.float32)
        y = np.random.randint(0, 10, (64,), dtype=np.int32)

        # Array implementation
        arr1 = Array.from_numpy(x)
        arr2 = Array.from_numpy(y)
        model = MnistModel()

        # PyTorch implementation
        torch_model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

        # Share weights between Array and PyTorch
        # First layer
        w1: torch.Tensor = model.linear1.w.torch()
        b1: torch.Tensor = model.linear1.b.torch()
        w1.requires_grad_(True)
        b1.requires_grad_(True)
        torch_model[0].weight.data.copy_(w1)
        torch_model[0].bias.data.copy_(b1)

        # Second layer
        w2: torch.Tensor = model.linear2.w.torch()
        b2: torch.Tensor = model.linear2.b.torch()
        w2.requires_grad_(True)
        b2.requires_grad_(True)
        torch_model[2].weight.data.copy_(w2)
        torch_model[2].bias.data.copy_(b2)

        # Forward pass
        t1 = torch.from_numpy(x)
        t2 = torch.from_numpy(y).type(torch.int64)
        model_jit = nn.Jit(model.__call__)
        loss_jit = nn.Jit(cross_entropy_loss)
        passes = 3

        for _ in range(passes):
            # JIT compilation here
            logits = model_jit(arr1)
            torch_logits = torch_model(t1)
            # Loss computation
            loss = loss_jit(logits, arr2)
            torch_loss: torch.Tensor = torch.nn.CrossEntropyLoss()(torch_logits, t2)
            # Backward pass
            loss.backward()
            torch_loss.backward()
            # Setup optimizers
            optimizer = GradientDescent(model.parameters(), lr=1e-3)
            torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-3, momentum=0, weight_decay=0)
            optimizer.step()
            torch_optimizer.step()
            # Compare updated weights and biases
            assert torch.allclose(model.linear1.w.torch(), torch_model[0].weight.data, atol=1e-3, rtol=0)
            assert torch.allclose(model.linear1.b.torch(), torch_model[0].bias.data, atol=1e-3, rtol=0)
            assert torch.allclose(model.linear2.w.torch(), torch_model[2].weight.data, atol=1e-3, rtol=0)
            assert torch.allclose(model.linear2.b.torch(), torch_model[2].bias.data, atol=1e-3, rtol=0)
