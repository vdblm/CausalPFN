"""Training utilities for the CausalPFN model.

The training stack is optional; install CausalPFN with the ``training``
extra before importing this package.
"""

from .trainer import calculate_loss, train

__all__ = ["calculate_loss", "train"]
