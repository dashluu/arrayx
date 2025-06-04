import arrayx.core


class Module:
    def __init__(self) -> None: ...

    def __call__(self, input: arrayx.core.Array) -> arrayx.core.Array:
        """Call the nn module using the forward hook"""

    def forward(self, input: arrayx.core.Array) -> arrayx.core.Array:
        """Forward the nn module, can be overidden"""

def relu(x: arrayx.core.Array) -> arrayx.core.Array:
    """ReLU activation function"""

def onehot(x: arrayx.core.Array, num_classes: int = -1) -> arrayx.core.Array:
    """One-hot encode input array"""

def cross_entropy_loss(x: arrayx.core.Array, y: arrayx.core.Array, num_classes: int = -1) -> arrayx.core.Array:
    """Compute cross-entropy loss between input x and target y"""
