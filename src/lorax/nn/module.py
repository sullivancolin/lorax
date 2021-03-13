"""
Abstract Module class with automatic Pytree registration
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import jax.numpy as np
from jax import jit
from jax.tree_util import register_pytree_node
from numpy import ndarray
from pydantic import BaseModel, PrivateAttr
from pydantic.utils import deep_update

from lorax.parameter import Parameter, ParamInit
from lorax.rng import RNG
from lorax.tensor import Tensor

__all__ = ["Module"]

T = TypeVar("T", bound="Module")

suffix = ".pkl"


class Module(BaseModel, ABC):
    _rng: Optional[RNG] = PrivateAttr(None)

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
            if field not in cls._params
            and (kind.name == "__root__" or issubclass(kind.type_, Module))
        ]
        cls._static = [
            field
            for field in cls.__fields__.keys()
            if field != "__root__"
            and field not in cls._params
            and field not in cls._children
        ]
        return super().__new__(cls)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if "_rng" in data:
            self._rng = data["_rng"]

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

        static = [getattr(self, val) for val in self._static] + [self._rng]

        keys = self._params + self._children + self._static + ["_rng"]
        return children, (keys, static)

    @classmethod
    def _tree_unflatten(
        cls, aux: Tuple[List[str], List[Any]], params: List[Union[Parameter, Module]]
    ) -> Module:
        keys = aux[0]
        vals = list(params) + aux[1]
        kwargs = {key: val for key, val in zip(keys, vals)}
        instance = cls.construct(**kwargs)  # type: ignore
        if "_rng" in kwargs:
            instance._rng = kwargs["_rng"]
        return instance

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        if self._rng is not None:
            return self.forward(inputs)
        raise ValueError("Module has not been initialized!")

    def update(self: T, **kwargs: Any) -> T:
        new_dict = deep_update(self.dict(), kwargs)
        new_instance = self.__class__(**new_dict)
        new_instance._rng = kwargs["_rng"]
        return new_instance

    def new_state(self: T, mode: str = "train") -> T:
        d: Dict[str, Any] = {"mode": mode, "_rng": self._rng}
        if mode == "train":
            new_rng, rng = self._rng.split()
            d["_rng"] = new_rng
        for m in self._children:
            new_m = getattr(self, m).new_state(mode)
            d[m] = new_m
        return self.update(**d)

    def initialize(self: T, rng: RNG) -> T:
        rng, new_rng = rng.split()
        d: Dict[str, Any] = {"_rng": new_rng}
        rng_p, rng_m = rng.split()
        rngs = rng_p.split(len(self._params))
        for name, rng in zip(self._params, rngs):
            p = getattr(self, name)
            if isinstance(p, ParamInit):
                d |= {name: p.instantiate(rng)}

        rngs = rng_m.split(len(self._children))
        for name, rng in zip(self._children, rngs):
            m = getattr(self, name)
            if m._rng is not None:
                d |= {name: m.initialize(rng)}

        return self.update(**d)

    def to_train(self: T) -> T:
        if self._rng is None:
            raise ValueError("Module has not been initialized!")
        return self.new_state(mode="train")

    def to_eval(self: T) -> T:
        if self._rng is None:
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
