from arrayx import Array
import ax
import numpy as np


with ax.context():
    arr1 = Array.zeros([2, 5, 3])
    arr2 = arr1[:, 1:4]
    arr2 += Array.ones([2, 3, 3])
    arr2.eval()
    print(arr1)
