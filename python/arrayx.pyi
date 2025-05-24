from collections.abc import Sequence
import enum

from numpy.typing import ArrayLike


class Dtype:
    def name(self) -> str:
        """Get dtype name as string"""

    def size(self) -> int:
        """Get size in bytes"""

    def __str__(self) -> str:
        """String representation of dtype"""

class F32(Dtype):
    """32-bit floating point dtype"""

class I32(Dtype):
    """32-bit integer dtype"""

class Bool(Dtype):
    """Boolean dtype"""

f32: F32 = ...

i32: I32 = ...

bool: Bool = ...

class Shape:
    @property
    def offset(self) -> int:
        """Get shape offset"""

    @property
    def view(self) -> list[int]:
        """Get shape view"""

    @property
    def stride(self) -> list[int]:
        """Get shape stride"""

    @property
    def ndim(self) -> int:
        """Get number of dimensions"""

    @property
    def numel(self) -> int:
        """Get total number of elements"""

    def __str__(self) -> str:
        """String representation of shape"""

class DeviceType(enum.Enum):
    CPU = 0

    MPS = 1

class Device:
    def type(self) -> DeviceType:
        """Get device type"""

    def id(self) -> int:
        """Get device ID"""

    def name(self) -> str:
        """Get device name"""

    def __str__(self) -> str:
        """String representation of device"""

class Backend:
    @staticmethod
    def init() -> None:
        """Initialize backend"""

    @staticmethod
    def cleanup() -> None:
        """Shutdown backend"""

class Array:
    @property
    def shape(self) -> Shape:
        """Get array shape"""

    @property
    def dtype(self) -> Dtype:
        """Get array data type"""

    @property
    def device(self) -> Device:
        """Get array device"""

    @property
    def ndim(self) -> int:
        """Get number of dimensions"""

    @property
    def numel(self) -> int:
        """Get total number of elements"""

    @property
    def offset(self) -> int:
        """Get array offset"""

    @property
    def view(self) -> list[int]:
        """Get array view"""

    @property
    def stride(self) -> list[int]:
        """Get array stride"""

    @property
    def ptr(self) -> int:
        """Get raw data pointer"""

    @property
    def itemsize(self) -> int:
        """Get size of each element in bytes"""

    @property
    def nbytes(self) -> int:
        """Get total size in bytes"""

    @property
    def is_contiguous(self) -> bool:
        """Check if array is contiguous"""

    def numpy(self) -> ArrayLike:
        """Convert array to numpy array"""

    def from_numpy(self) -> Array:
        """Convert numpy array to array"""

    @staticmethod
    def full(view: Sequence[int], value: int, dtype: Dtype = ..., device: str = 'cpu') -> Array:
        """Create a new array filled with specified value"""

    @staticmethod
    def zeros(view: Sequence[int], dtype: Dtype = ..., device: str = 'cpu') -> Array:
        """Create a new array filled with zeros"""

    @staticmethod
    def ones(view: Sequence[int], dtype: Dtype = ..., device: str = 'cpu') -> Array:
        """Create a new array filled with ones"""

    @staticmethod
    def arange(view: Sequence[int], start: int, step: int, dtype: Dtype = ..., device: str = 'cpu') -> Array:
        """Create a new array with evenly spaced values"""

    @staticmethod
    def zeros_like(other: Array, dtype: Dtype = ..., device: str = 'cpu') -> Array:
        """Create a new array of zeros with same shape as input"""

    @staticmethod
    def ones_like(other: Array, dtype: Dtype = ..., device: str = 'cpu') -> Array:
        """Create a new array of ones with same shape as input"""

    def __add__(self, rhs: Array) -> Array:
        """Add two arrays element-wise"""

    def __sub__(self, rhs: Array) -> Array:
        """Subtract two arrays element-wise"""

    def __mul__(self, rhs: Array) -> Array:
        """Multiply two arrays element-wise"""

    def __truediv__(self, rhs: Array) -> Array:
        """Divide two arrays element-wise"""

    def __iadd__(self, rhs: Array) -> Array:
        """In-place add two arrays element-wise"""

    def __isub__(self, rhs: Array) -> Array:
        """In-place subtract two arrays element-wise"""

    def __imul__(self, rhs: Array) -> Array:
        """In-place multiply two arrays element-wise"""

    def __itruediv__(self, rhs: Array) -> Array:
        """In-place divide two arrays element-wise"""

    def exp(self, in_place: bool = False) -> Array:
        """Compute exponential of array elements"""

    def log(self, in_place: bool = False) -> Array:
        """Compute natural logarithm of array elements"""

    def sqrt(self, in_place: bool = False) -> Array:
        """Compute square root of array elements"""

    def sq(self, in_place: bool = False) -> Array:
        """Compute square of array elements"""

    def neg(self, in_place: bool = False) -> Array:
        """Compute negative of array elements"""

    def recip(self, in_place: bool = False) -> Array:
        """Compute reciprocal of array elements"""

    def eq(self, rhs: Array) -> Array:
        """Element-wise equality comparison"""

    def neq(self, rhs: Array) -> Array:
        """Element-wise inequality comparison"""

    def lt(self, rhs: Array) -> Array:
        """Element-wise less than comparison"""

    def gt(self, rhs: Array) -> Array:
        """Element-wise greater than comparison"""

    def leq(self, rhs: Array) -> Array:
        """Element-wise less than or equal comparison"""

    def geq(self, rhs: Array) -> Array:
        """Element-wise greater than or equal comparison"""

    def sum(self, dims: Sequence[int] = []) -> Array:
        """Sum array elements along specified dimensions"""

    def max(self, dims: Sequence[int] = []) -> Array:
        """Maximum value along specified dimensions"""

    def min(self, dims: Sequence[int] = []) -> Array:
        """Minimum value along specified dimensions"""

    def argmax(self, dims: Sequence[int] = []) -> Array:
        """Indices of maximum values along specified dimensions"""

    def argmin(self, dims: Sequence[int] = []) -> Array:
        """Indices of minimum values along specified dimensions"""

    def broadcast(self, view: Sequence[int]) -> Array:
        """Broadcast array to new shape"""

    def broadcast_to(self, view: Sequence[int]) -> Array:
        """Broadcast array to target shape"""

    def reshape(self, view: Sequence[int]) -> Array:
        """Reshape array to new dimensions"""

    def flatten(self, start_dim: int, end_dim: int) -> Array:
        """Flatten dimensions from start to end"""

    def squeeze(self, dim: int) -> Array:
        """Remove single-dimensional entry from array"""

    def unsqueeze(self, dim: int) -> Array:
        """Add single-dimensional entry to array"""

    def permute(self, dims: Sequence[int]) -> Array:
        """Permute array dimensions"""

    def transpose(self, start_dim: int = 0, end_dim: int = 1) -> Array:
        """Transpose array dimensions"""

    def astype(self, dtype: Dtype) -> Array:
        """Cast array to specified dtype"""

    def eval(self) -> None:
        """Evaluate array and materialize values"""

    def backward(self) -> None:
        """Compute gradients through backpropagation"""

    def __str__(self) -> str:
        """String representation of array"""
