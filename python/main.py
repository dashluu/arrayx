from arrayx import Array, f32
import ax
import numpy as np
import torch


with ax.context():
    x = torch.randn(2, 3, dtype=torch.float32)
    arr1 = Array.from_numpy(x.numpy())
    arr2 = arr1.mean([0])
    arr3 = (arr1 > 0).astype(f32)
    arr3.eval()
    print(arr1)
    print(arr3)
