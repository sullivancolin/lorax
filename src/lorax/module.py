"""
Abstract Module class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
from __future__ import annotations
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union

import jax.numpy as np
from jax import jit
from jax.tree_util import register_pytree_node
from numpy import ndarray
from pydantic import BaseModel

from lorax.parameter import Parameter
from lorax.tensor import Tensor

__all__ = ["Module"]

T = TypeVar("T", bound="Module")

suffix = ".pkl"


class Module(BaseModel, ABC):
    class Config:
        allow_mutation = False
        # frozen = True
        json_encoders = {
            np.DeviceArray: lambda t: f"shape={t.shape}",
            ndarray: lambda a: f"shape={a.shape}",
        }

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        if not is_abstract:
            register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)

    def __new__(cls, *args: Any, **kwargs: Any) -> Module:
        cls._params = [
            field
            for field, kind in cls.__fields__.items()
            if kind.name != "__root__" and issubclass(kind.type_, Parameter)
        ]

        cls._children = [
            field
            for field, kind in cls.__fields__.items()
            if kind.name == "__root__" or issubclass(kind.type_, Module)
        ]
        cls._static = [
            field
            for field in cls.__fields__.keys()
            if field != "__root__"
            and field not in cls._params
            and field not in cls._children
        ]
        return super().__new__(cls)

    def json(self, *args: Any, **kwargs: Any) -> str:
        return super().json(indent=4)

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            **super().dict(),
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return self.json()

    def __str__(self) -> str:
        return self.json()

    def __hash__(self) -> int:
        return id(self)

    def _tree_flatten(
        self,
    ) -> Tuple[List[Union[Parameter, Module]], Tuple[List[str], List[Any]]]:
        children = [getattr(self, val) for val in self._params]

        children += [getattr(self, val) for val in self._children]

        static = [getattr(self, val) for val in self._static]

        keys = self._params + self._children + self._static
        return children, (keys, static)

    @classmethod
    def _tree_unflatten(
        cls, aux: Tuple[List[str], List[Any]], params: List[Union[Parameter, Module]]
    ) -> "Module":
        keys = aux[0]
        vals = list(params) + aux[1]
        kwargs = {key: val for key, val in zip(keys, vals)}
        return cls.construct(**kwargs)  # type: ignore

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def children(self) -> Iterator[Module]:
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        for name in self._children:
            module = getattr(self, name)
            yield name, module

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self, memo: Optional[Set["Module"]] = None, prefix: str = ""
    ) -> Iterator[Tuple[str, "Module"]]:

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name in self._children:
                module = getattr(self, name)
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        path = Path(path)
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f"File {path} already exists.")
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            instance = pickle.load(file)
        return instance
