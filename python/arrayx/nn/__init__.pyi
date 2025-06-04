import arrayx.core


class Module:
    def __call__(self, input: arrayx.core.Array) -> arrayx.core.Array:
        """Call the nn module using the forward hook"""

    def forward(self, input: arrayx.core.Array) -> arrayx.core.Array:
        """Forward the nn module, can be overidden"""
