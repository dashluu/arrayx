from arrayx.core import Array, f32, b8
import nn
import ax
import numpy as np
import torch


with ax.context():
    # x = torch.randn(2, 3, dtype=torch.float32)
    # arr1 = Array.from_numpy(x.numpy())
    # arr2 = arr1.mean()
    # arr3 = (arr1 > 0).astype(f32)
    # print(arr1)
    # print(arr3)
    # arr4 = arr2.astype(b8)
    # print(arr2.item())
    # print(arr4.item())
    x = np.random.randn(2, 3).astype(np.float32)
    arr1 = Array.from_numpy(x)
    t1 = torch.from_numpy(x)
    linear = nn.Linear(3, 4)
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
