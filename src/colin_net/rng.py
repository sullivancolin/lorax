from typing import Any, Dict, Tuple, Union

import jax.numpy as np
from jax import random
from numpy import ndarray

from colin_net.module import Module
from colin_net.tensor import Tensor


class RNG(Module):
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

    def static_params(self) -> Dict[str, Any]:
        return {"int_1": self.int_1, "int_2": self.int_2}

    def trainable_params(self) -> Dict[str, Any]:
        return {}
