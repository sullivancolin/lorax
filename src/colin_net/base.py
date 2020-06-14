"""
Abstract Base class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
from typing import Any, List, Tuple, Hashable, Dict, Union

from jax.tree_util import register_pytree_node
from jax.interpreters.ad import JVPTracer
from jax import random
from pydantic import BaseModel
from numpy import ndarray
import jax.numpy as np

from colin_net.tensor import Tensor

__all__ = ["Module"]


FlattenedParams = Tuple[Union["Module", Tensor], ...]
FlattenedRemainder = Tuple[Tuple[str, Hashable], ...]
FlattenedMetadata = Tuple[str, ...]
FlattenedAux = Tuple[FlattenedRemainder, FlattenedMetadata]


class Module(BaseModel):
    differentiable: bool = True

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

    def _normal_dict(self) -> Dict[str, Any]:
        return dict(
            self._iter(
                to_dict=True,
                by_alias=False,
                include=None,
                exclude=None,
                exclude_unset=False,
                exclude_defaults=False,
                exclude_none=False,
            )
        )

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {self.__class__.__name__: super().dict()}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self.json()

    def replace(self, **updates: Any) -> BaseModel:
        self_dict = self._normal_dict()
        self_dict.update(updates)
        return self.__class__(**self_dict)

    def tree_flatten(
        self,
    ) -> Tuple[
        FlattenedParams, FlattenedAux,
    ]:
        fields = self.__fields__
        params = []
        param_metadata = []
        if self.differentiable is False:
            return (), (tuple((key, getattr(self, key)) for key in fields), ())  # type: ignore
        for key in fields:
            val = getattr(self, key)
            if (
                isinstance(val, Tensor)
                or isinstance(val, Module)
                or isinstance(val, JVPTracer)
            ):
                params.append(val)
                param_metadata.append(key)
            elif isinstance(val, List) and isinstance(val[0], Module):
                params.append(val)
                param_metadata.append(key)

        flattened_remainder = tuple(
            (key, getattr(self, key)) for key in fields if key not in param_metadata
        )
        flattened_metadata = tuple(param_metadata)

        return tuple(params), (flattened_remainder, flattened_metadata)  # type: ignore

    @classmethod
    def tree_unflatten(cls, aux: FlattenedAux, params: FlattenedParams,) -> "Module":
        # breakpoint()
        flattened_remainder, flattened_metadata = aux

        metadata_dict = {key: val for (key, val) in flattened_remainder}

        param_dict = {key: val for (key, val) in zip(flattened_metadata, params)}

        constructor_dict = {**param_dict, **metadata_dict}
        if "key" in constructor_dict:
            rngwrapper: RNGWrapper = constructor_dict["key"]

            newwrapper = RNGWrapper.generate(rngwrapper)
            constructor_dict["key"] = newwrapper

        # Disable validation from unflattening for speed up
        return cls.construct(**constructor_dict)  # type: ignore


class RNGWrapper(Module):
    int_1: int
    int_2: int

    def to_prng(self) -> Tensor:
        return np.array([self.int_1, self.int_2], dtype=np.uint32)

    @classmethod
    def from_prng(cls, key: Union[ndarray, Tensor]) -> "RNGWrapper":
        return cls(int_1=key[0], int_2=key[1])

    @classmethod
    def generate(cls, other: "RNGWrapper") -> "RNGWrapper":
        orig_key = np.array([other.int_1, other.int_2], dtype=np.uint32)
        key, subkey = random.split(orig_key)
        return cls(int_1=subkey[0], int_2=subkey[1])
