from arrayx import Array
import ax
import numpy as np


with ax.context():
    arr1 = Array.ones([2, 1, 5, 1, 6])
    arr2 = arr1.squeeze([1, 3])
    arr2 += 1
    arr3 = arr2 < 0
    arr4 = arr2.mean()
    arr4.eval()
    print(arr2.shape)
    print(arr2)
    print(arr4)
