"""
A tensor is just a n-dimensional array
"""
from jax.interpreters.xla import DeviceArray as Tensor

__all__ = ["Tensor"]
