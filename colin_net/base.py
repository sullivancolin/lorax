"""
Abstract Base class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
from abc import abstractmethod

from jax.tree_util import register_pytree_node

__all__ = ["PyTreeLike"]


class PyTreeLike:
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data, values):
        """
        Args:
            aux_data:
                Data that will be treated as constant through JAX operations.
            values:
                A JAX pytree of values from which the object is constructed.
        Returns:
            A constructed object.
        """
        raise NotImplementedError

    @abstractmethod
    def tree_flatten(self):
        """
        Returns:
            values: A JAX pytree of values representing the object.
            aux_data:
                Data that will be treated as constant through JAX operations.
        """
        raise NotImplementedError
