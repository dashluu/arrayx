import arrayx.core


class Module:
    def __init__(self) -> None: ...

    def __call__(self, x: arrayx.core.Array) -> arrayx.core.Array:
        """Call the nn module using the forward hook"""

    def forward(self, x: arrayx.core.Array) -> arrayx.core.Array:
        """Forward the nn module, can be overidden"""

    def parameters(self) -> list[arrayx.core.Array]:
        """Get the parameters of the nn module, can be overidden"""

    def jit(self, x: arrayx.core.Array) -> arrayx.core.Array:
        """JIT-compile the nn module"""

def linear(x: arrayx.core.Array, weight: arrayx.core.Array) -> arrayx.core.Array:
    """Functional linear without bias"""

def linear_with_bias(x: arrayx.core.Array, weight: arrayx.core.Array, bias: arrayx.core.Array) -> arrayx.core.Array:
    """Functional linear with bias"""

def relu(x: arrayx.core.Array) -> arrayx.core.Array:
    """ReLU activation function"""

def onehot(x: arrayx.core.Array, num_classes: int = -1) -> arrayx.core.Array:
    """One-hot encode input array"""

def cross_entropy_loss(x: arrayx.core.Array, y: arrayx.core.Array) -> arrayx.core.Array:
    """Compute cross-entropy loss between input x and target y"""
