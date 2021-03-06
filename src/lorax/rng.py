from typing import Any, Dict, Tuple, Union

import jax.numpy as np
from jax import random
from jax.tree_util import register_pytree_node
from numpy import ndarray
from pydantic import BaseModel

from lorax.tensor import Tensor


class RNG(BaseModel):
    int_1: int
    int_2: int

    def to_prng(self) -> Tensor:
        return np.array([self.int_1, self.int_2], dtype=np.uint32)

    @classmethod
    def from_seed(self, seed: int) -> "RNG":
        key = random.PRNGKey(seed)
        return RNG(int_1=key[0], int_2=key[1])

    @classmethod
    def from_prng(cls, key: Union[ndarray, Tensor]) -> "RNG":
        return cls(int_1=key[0], int_2=key[1])

    def split(self, num: int = 2) -> "Tuple[RNG, ...]":
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


def _flatten_rng(rng: RNG) -> Tuple[Tuple[int, int], None]:
    return (rng.int_1, rng.int_2), None


def _unflatten_rng(aux: Any, params: Tuple[int, int]) -> RNG:
    return RNG.construct(int_1=params[0], int_2=params[1])


register_pytree_node(RNG, _flatten_rng, _unflatten_rng)
