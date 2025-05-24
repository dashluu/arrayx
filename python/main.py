from arrayx import Array, f32, Backend
import ax
import numpy as np

with ax.context():
    np1 = np.random.randn(2, 3, 4).astype(np.float32)
    np2 = np.random.randn(2, 3, 4).astype(np.float32)
    arr1 = Array.from_numpy(np1)
    arr2 = Array.from_numpy(np2)
    arr3: Array = arr1 + arr2
    arr3.eval()
    print(arr1)
    print(arr2)
    print(arr3)
