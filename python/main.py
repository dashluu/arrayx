from arrayx import Array, f32, b8
import ax
import numpy as np
import torch


with ax.context():
    x = torch.randn(2, 3, dtype=torch.float32)
    arr1 = Array.from_numpy(x.numpy())
    arr2 = arr1.mean()
    arr3 = (arr1 > 0).astype(f32)
    arr3.eval()
    print(arr1)
    print(arr3)
    arr2.eval()
    arr4 = arr2.astype(b8)
    arr4.eval()
    print(arr2.item())
    print(arr4.item())
