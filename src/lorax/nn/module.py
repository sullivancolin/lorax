"""
Abstract Module class with automatic Pytree registration
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypeVar

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
    # _is_initialized: bool = PrivateAttr(False)
    _module_registry: dict[str, Type[Module]] = {}

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
            if cls.__name__.lower() in cls._module_registry:
                raise ValueError(f"{cls.__name__.lower()} has already been registered.")
            cls._module_registry[cls.__name__.lower()] = cls

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value: dict[str, Any]) -> Module:
        try:
            sub_class = value["name"]
            return cls._module_registry[sub_class](**value)
        except Exception as e:
            raise ValueError("No registered subclass.") from e

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
            and (
                kind.name == "__root__"
                or kind.name == "name"
                or issubclass(kind.type_, Module)
            )
        ]
        cls._static = [
            field
            for field in cls.__fields__.keys()
            if field != "__root__"
            and field not in cls._params
            and field not in cls._children
        ]
        cls.name = cls.__name__.lower()
        return super().__new__(cls)

    # def __init__(self, **data: Any) -> None:
    #     super().__init__(**data)
    #     if "_is_initialized" in data:
    #         self._is_initialized = data["_is_initialized"]

    def json(self, *args: Any, **kwargs: Any) -> str:
        return super().json(indent=4)

    # def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    #     return {"type": self.__class__.__name__, **super().dict()}

    # def __repr__(self) -> str:
    #     return self.json()

    def __str__(self) -> str:
        return self.json()

    def __hash__(self) -> int:
        return id(self)

    def _tree_flatten(
        self,
    ) -> tuple[list[Parameter | Module], tuple[list[str], list[Any]]]:
        children = [getattr(self, val) for val in self._params]

        children += [getattr(self, val) for val in self._children]

        static = [getattr(self, val) for val in self._static] + [self._is_initialized]

        keys = self._params + self._children + self._static + ["_rng"]
        return children, (keys, static)

    @classmethod
    def _tree_unflatten(
        cls, aux: tuple[list[str], list[Any]], params: list[Parameter | Module]
    ) -> Module:
        keys = aux[0]
        vals = list(params) + aux[1]
        kwargs: dict[str, Any] = {key: val for key, val in zip(keys, vals)}
        instance = cls.construct(**kwargs)  # type: ignore
        if "_is_initialized" in kwargs:
            instance._is_initialized = kwargs["_is_initialized"]
        return instance

    # @abstractmethod
    # def forward(self, inputs: Tensor) -> Tensor:
    #     raise NotImplementedError

    # @jit
    # def __call__(self, inputs: Tensor) -> Tensor:
    #     if self._is_initialized:
    #         return self.forward(inputs)
    #     raise ValueError("Module has not been initialized!")

    # def update(self: T, **kwargs: Any) -> T:
    #     new_dict = deep_update(self.dict(), kwargs)
    #     new_instance = self.__class__(**new_dict)
    #     new_instance._is_initialized = kwargs["_is_initialized"]  # type: ignore
    #     return new_instance

    # def new_state(self: T, **kwargs: Any) -> T:
    #     d = {}
    #     for m in self._children:
    #         new_m = getattr(self, m).new_state(**kwargs)
    #         d[m] = new_m
    #     d |= kwargs
    #     return self.update(**d)

    # # def to_train(self: T) -> T:
    # #     if not self._is_initialized:
    # #         raise ValueError("Module has not been initialized!")
    # #     if hasattr(self, "rng"):
    # #         rng, iter_rng = self.rng.split()
    # #         return self.new_state(mode="train", rng=rng)
    # #     return self.new_state(mode="train")

    # # def to_eval(self: T) -> T:
    # #     if not self._is_initialized:
    # #         raise ValueError("Module has not been initialized!")
    # #     return self.new_state(mode="eval")

    # def initialize(self: T, rng: RNG) -> T:
    #     rng, new_rng = rng.split()
    #     d: dict[str, Any] = {"rng": new_rng}
    #     rng_p, rng_m = rng.split()
    #     rngs = rng_p.split(len(self._params))
    #     for name, rng in zip(self._params, rngs):
    #         p = getattr(self, name)
    #         if isinstance(p, ParamInit):
    #             d |= {name: p.init(rng)}

    #     rngs = rng_m.split(len(self._children))
    #     for name, rng in zip(self._children, rngs):
    #         m = getattr(self, name)
    #         if m._rng is not None:
    #             d |= {name: m.initialize(rng)}

    #     return self.update(**d)

    # def save(self, path: str | Path, overwrite: bool = False) -> None:
    #     path = Path(path)
    #     if path.suffix != suffix:
    #         path = path.with_suffix(suffix)
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     if path.exists():
    #         if overwrite:
    #             path.unlink()
    #         else:
    #             raise RuntimeError(f"File {path} already exists.")
    #     with open(path, "wb") as file:
    #         pickle.dump(self, file)

    # @classmethod
    # def load(cls: Type[T], path: str | Path) -> T:
    #     path = Path(path)
    #     if not path.is_file():
    #         raise ValueError(f"Not a file: {path}")
    #     if path.suffix != suffix:
    #         raise ValueError(f"Not a {suffix} file: {path}")
    #     with open(path, "rb") as file:
    #         instance = pickle.load(file)
    #     return instance
