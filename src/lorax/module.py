"""
Abstract Base class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple

import jax.numpy as np
from jax.tree_util import register_pytree_node
from numpy import ndarray
from pydantic import BaseModel

__all__ = ["Module"]


class Module(BaseModel, ABC):
    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        if not is_abstract:
            register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    class Config:
        allow_mutation = False
        json_encoders = {
            np.DeviceArray: lambda t: f"shape={t.shape}",
            ndarray: lambda a: f"shape={a.shape}",
        }

    def json(self, *args: Any, **kwargs: Any) -> str:
        return super().json(indent=4)

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {self.__class__.__name__: super().dict()}

    def __repr__(self) -> str:
        return self.json()

    def __str__(self) -> str:
        return self.json()

    @abstractmethod
    def trainable_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def static_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def tree_flatten(
        self,
    ) -> Tuple[Iterable[Any], Tuple[Iterable[str], Iterable[Tuple[str, Any]]]]:
        children = list(self.trainable_params().values())
        keys = list(self.trainable_params().keys())
        return children, (keys, self.static_params().items())

    @classmethod
    def tree_unflatten(
        cls,
        aux: Tuple[Iterable[str], Iterable[Tuple[str, Any]]],
        params: Iterable[Any],
    ) -> "Module":
        keys, meta_data = aux
        d = dict(zip(keys, params))
        d.update(dict(meta_data))
        return cls.construct(**d)
