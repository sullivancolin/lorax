"""
Abstract Base class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, Tuple, Union

import jax.numpy as np
from jax import jit, random
from jax.tree_util import register_pytree_node
from numpy import ndarray
from pydantic import BaseModel

from colin_net.tensor import Tensor

__all__ = ["Module", "RNGWrapper"]


class Module(BaseModel):
    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        if not is_abstract:
            register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {
            Tensor: lambda t: f"shape={t.shape}",
            ndarray: lambda a: f"shape={a.shape}",
        }

    def json(self, *args: Any, **kwargs: Any) -> str:
        return super().json(indent=4)

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {self.__class__.__name__: super().dict()}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self.json()

    @abstractmethod
    def tree_flatten(self) -> Any:
        raise NotImplementedError

    @abstractclassmethod
    def tree_unflatten(cls, aux: Any, params: Any) -> "Module":
        raise NotImplementedError


class RNGWrapper(Module):
    int_1: int
    int_2: int

    def to_prng(self) -> Tensor:
        return np.array([self.int_1, self.int_2], dtype=np.uint32)

    @classmethod
    def from_prng(cls, key: Union[ndarray, Tensor]) -> "RNGWrapper":
        return cls(int_1=key[0], int_2=key[1])

    @jit
    def split(self) -> "RNGWrapper":
        orig_key = np.array([self.int_1, self.int_2], dtype=np.uint32)
        key, subkey = random.split(orig_key)
        return RNGWrapper(int_1=subkey[0], int_2=subkey[1])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RNGWrapper):
            return NotImplemented
        return self.int_1 == other.int_1 and self.int_2 == other.int_2

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, RNGWrapper):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.int_1, self.int_2))

    def tree_flatten(self) -> Tuple[Tuple[None], Tuple[int, int]]:
        return (None,), (self.int_1, self.int_2)

    @classmethod
    def tree_unflatten(cls, aux: Tuple[int, int], params: Tuple[None]) -> "RNGWrapper":
        return cls.construct(int_1=aux[0], int_2=aux[1])
