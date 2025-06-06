from collections.abc import Sequence

import arrayx.core


class Optimizer:
    def __init__(self, params: Sequence[arrayx.core.Array], lr: float = 0.001) -> None:
        """Base optimizer"""

    def forward(self) -> None:
        """Parameter update function"""

    def step(self) -> None:
        """Update module parameters"""

class GradientDescent(Optimizer):
    def __init__(self, params: Sequence[arrayx.core.Array], lr: float = 0.001) -> None:
        """Gradient Descent optimizer"""
