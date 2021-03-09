"""
Abstract Module class with automatic Pytree registration
Inspired by from https://github.com/google/jax/issues/2916
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import jax.numpy as np
from jax import jit, random
from jax.tree_util import register_pytree_node
from numpy import ndarray
from pydantic import BaseModel, PrivateAttr
from pydantic.utils import deep_update

from lorax.parameter import Parameter, ParamInit
from lorax.tensor import Tensor

__all__ = ["Module"]

T = TypeVar("T", bound="Module")

suffix = ".pkl"


class Module(BaseModel, ABC):
    _is_initialized: bool = False
    # _rng: Optional[Tensor] = PrivateAttr(False)

    class Config:
        allow_mutation = False
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
            if kind.name != "__root__" and kind.type_ == Parameter
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

    # def __init__(self, **data: Any) -> None:
    #     super().__init__(**data)
    #     if "_rng" in data:
    #         self._rng = data["_rng"]
    #     if "_is_initialized" in data:
    #         self._is_initialized = data["_is_initialized"]

    def json(self, *args: Any, **kwargs: Any) -> str:
        return super().json(indent=4)

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, **super().dict()}

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
        if self.is_initialized:
            return self.forward(inputs)
        raise ValueError("Module has not been initialized!")

    def update(self: T, **kwargs: Any) -> T:
        new_dict = deep_update(self.dict(), kwargs)
        return self.__class__(**new_dict)

    def new_state(self: T, mode: str = "train") -> T:
        d: Dict[str, Any] = {"mode": mode}
        for m in self._children:
            new_m = getattr(self, m).new_state(mode)
            d[m] = new_m
        return self.update(**d)

    def initialize(self: T, rng: Tensor) -> T:
        d: Dict[str, Any] = {"rng": rng, "_is_initialized": True}
        rng_p, rng_m = random.split(rng)
        rngs = random.split(rng_p, len(self._params))
        for name, rng in zip(self._params, rngs):
            p = getattr(self, name)
            if isinstance(p, ParamInit):
                d |= {name: p.instantiate(rng)}

        rngs = random.split(rng_m, len(self._children))
        for name, rng in zip(self._children, rngs):
            m = getattr(self, name)
            if not m._is_initialized:
                d |= {name: m.initialize(rng)}

        return self.update(**d)

    def to_train(self) -> T:
        if not self.is_initialized:
            raise ValueError("Module has not been initialized!")
        return self.new_state(mode="train")

    def to_eval(self) -> T:
        if not self._is_initialized:
            raise ValueError("Module has not been initialized!")
        return self.new_state(mode="eval")

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
