import arrayx.core


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
