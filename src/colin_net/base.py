"""
Abstract Base class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
import json
from typing import Any, List, Tuple, Union

from jax.tree_util import register_pytree_node
from jax.interpreters.ad import JVPTracer
from pydantic import BaseModel

from colin_net.tensor import Tensor

__all__ = ["Module"]


FlattenedParams = Tuple[Union["Module", Tensor], ...]
FlattenedMetadata = Tuple[Tuple[Tuple[str, Any]], Tuple[Tuple[str, ...]]]


class Module(BaseModel):
    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        if not is_abstract:
            register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {Tensor: lambda t: f"shape={t.shape}"}

    def json(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps({self.__class__.__name__: json.loads(super().json())})

    def __repr__(self) -> str:
        return self.json()

    def tree_flatten(
        self,
    ) -> Tuple[
        FlattenedParams, FlattenedMetadata,
    ]:
        fields = self.__fields__
        params = []
        param_metadata = []
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
    def tree_unflatten(
        cls, aux: FlattenedMetadata, params: FlattenedParams,
    ) -> "Module":
        # breakpoint()
        flattened_remainder, flattened_metadata = aux

        metadata_dict = {key: val for (key, val) in flattened_remainder}

        param_dict = {key: val for (key, val) in zip(flattened_metadata, params)}

        constructor_dict = {**param_dict, **metadata_dict}  # type: ignore

        # Disable validation from unflattening for speed up
        return cls.construct(**constructor_dict)  # type: ignore
