from arrayx import Array, f32, Backend
import ax
import numpy as np
import torch


def compare_grads(arr_grad: Array, torch_grad: torch.Tensor, name: str):
    arr_np = arr_grad.numpy()
    torch_np = torch_grad.detach().numpy()

    # Find indices where values differ more than tolerance
    mask = ~np.isclose(arr_np, torch_np, atol=1e-3, rtol=0)
    if np.any(mask):
        mismatch_indices = np.where(mask)
        print(f"\nFirst mismatch for {name}:")
        print("Index | Array Grad | Torch Grad | Difference")
        print("-" * 50)
        # Get only the first mismatched index
        idx = tuple(i[0] for i in mismatch_indices)
        arr_val = arr_np[idx]
        torch_val = torch_np[idx]
        diff = abs(arr_val - torch_val)
        print(f"{idx} | {arr_val:.6f} | {torch_val:.6f} | {diff:.6f}")
        raise AssertionError(f"Gradient mismatched for {name}")


with ax.context():
    np1 = np.random.uniform(0.1, 2.0, size=(2, 3)).astype(np.float32)
    arr1 = Array.from_numpy(np1)
    arr2 = arr1.exp()
    arr3 = arr2 * arr1
    arr4 = arr3.log()
    arr5 = arr4 / arr1
    arr6 = arr5.sum()
    arr6.eval()
    arr6.backward()
    t1 = torch.from_numpy(np1).requires_grad_(True)
    t2 = torch.exp(t1)
    t2.retain_grad()
    t3 = t2 * t1
    t3.retain_grad()
    t4 = torch.log(t3)
    t4.retain_grad()
    t5 = t4 / t1
    t5.retain_grad()
    t6 = t5.sum()
    t6.backward()
    print(arr6)
    compare_grads(arr5.grad, t5.grad, "arr5")
    compare_grads(arr4.grad, t4.grad, "arr4")
    compare_grads(arr3.grad, t3.grad, "arr3")
    compare_grads(arr2.grad, t2.grad, "arr2")
    compare_grads(arr1.grad, t1.grad, "arr1")
