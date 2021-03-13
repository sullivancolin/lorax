from __future__ import annotations

from typing import Any, Tuple, TypeVar, Union

import jax.numpy as np
from jax import random
from numpy import ndarray
from pydantic import BaseModel

from lorax.tensor import Tensor

T = TypeVar("T", bound="RNG")


class RNG(BaseModel):
    int_1: int
    int_2: int

    # def __init_subclass__(cls, **kwargs: Any) -> None:
    #     super().__init_subclass__(**kwargs)  # type: ignore
    #     register_pytree_node(cls, cls._flatten_rng, cls._unflatten_rng)

    def to_prng(self) -> Tensor:
        return np.array([self.int_1, self.int_2], dtype=np.uint32)

    @classmethod
    def from_seed(self, seed: int) -> RNG:
        key = random.PRNGKey(seed)
        return RNG(int_1=key[0], int_2=key[1])

    @classmethod
    def from_prng(cls, key: Union[ndarray, Tensor]) -> RNG:
        return cls(int_1=key[0], int_2=key[1])

    def split(self, num: int = 2) -> Tuple[RNG, ...]:
        orig_key = np.array([self.int_1, self.int_2], dtype=np.uint32)
        new_keys = random.split(orig_key, num=num)

        return tuple(RNG.from_prng(key) for key in new_keys)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RNG):
            return NotImplemented
        return self.int_1 == other.int_1 and self.int_2 == other.int_2

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, RNG):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.int_1, self.int_2))

    # def _flatten_rng(self) -> Tuple[None, Tuple[int, int]]:
    #     return None, (self.int_1, self.int_2)

    # @classmethod
    # def _unflatten_rng(cls: Type[T], aux: Tuple[int, int], params: Any) -> T:
    #     return cls.construct(int_1=aux[0], int_2=aux[1])
