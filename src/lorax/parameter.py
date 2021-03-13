from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Union

from pydantic import BaseModel

from lorax.nn.functional import INITIALIZERS, InitializerEnum
from lorax.rng import RNG
from lorax.tensor import Tensor


class ParamInit(BaseModel):

    shape: Tuple[int, int]
    initializer: InitializerEnum = InitializerEnum.normal

    class Config:
        allow_mutation = False

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            **super().dict(),
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"Parameter shape={self.shape}"

    def __str__(self) -> str:
        return self.json()

    def instantiate(
        self,
        rng: RNG,
        initializer: Callable[..., Tensor] = None,
    ) -> Tensor:
        """Returns a tensor created according to this init."""
        if initializer is not None:
            return initializer(key=rng.to_prng(), shape=self.shape)
        return INITIALIZERS[self.initializer](key=rng.to_prng(), shape=self.shape)


Parameter = Union[Tensor, ParamInit]
