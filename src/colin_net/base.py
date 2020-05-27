"""
Abstract Base class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
import json
from typing import Any, Dict, List, Tuple, Union

from jax.tree_util import register_pytree_node
from pydantic import BaseModel

from colin_net.tensor import Tensor

__all__ = ["Module"]


FlattenedParams = Tuple[Union["Module", Tensor], ...]
FlattenedMetadata = Tuple[List[Tuple[str, Any]], List[Tuple[str, int]]]


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
        d = self.__dict__
        flattened_params = []
        param_metadata = {}

        for key, val in d.items():
            if isinstance(val, Tensor) or isinstance(val, Module):
                flattened_params.append(val)
                param_metadata[key] = 0
            elif isinstance(val, List) and isinstance(val[0], Module):
                flattened_params.extend(val)
                param_metadata[key] = len(val)

        flattened_remainder = [
            (key, val) for key, val in d.items() if key not in param_metadata
        ]
        flattened_metadata = [(key, val) for key, val in param_metadata.items()]
        return tuple(flattened_params), (flattened_remainder, flattened_metadata)

    @classmethod
    def tree_unflatten(
        cls, aux: FlattenedMetadata, params: FlattenedParams,
    ) -> "Module":
        flattened_remainder, flattened_metadata = aux
        flattened_params = list(params)
        metadata_dict: Dict[str, int] = {key: val for (key, val) in flattened_metadata}
        constructor_dict: Dict[str, Any] = {
            key: val for (key, val) in flattened_remainder
        }
        for key, val in metadata_dict.items():
            if val == 0:
                constructor_dict[key] = flattened_params.pop(0)
            else:
                items = []
                for _ in range(val):
                    items.append(flattened_params.pop(0))
                constructor_dict[key] = items

        # Disable validation from unflattening for speed up
        return cls.construct(**constructor_dict)
