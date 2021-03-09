"""
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterator, Optional, Tuple

import jax.numpy as np
from jax import jit, random

from lorax.nn import Module
from lorax.nn.functional import ACTIVATIONS, ActivationEnum, InitializerEnum
from lorax.parameter import Parameter, ParamInit

# from lorax.rng import RNG
from lorax.tensor import Tensor


class Mode(str, Enum):
    """Allowed values for Layers with different behaviors during training and inference."""

    train = "train"
    eval = "eval"


class Linear(Module):
    """Dense Linear Layer.
    Computes output = activation(inputs @ w + b)"""

    w: Parameter
    b: Parameter
    activation: ActivationEnum = ActivationEnum.identity

    @jit
    def forward(self, inputs: Tensor) -> Tensor:
        """outputs = self.activation(inputs * self.w + self.b)"""

        return ACTIVATIONS[self.activation](inputs @ self.w + self.b)

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        activation: ActivationEnum = ActivationEnum.identity,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> Linear:
        """Factory for new Linear from input and output dimensions"""
        return cls(
            w=ParamInit(shape=(input_dim, output_dim), initializer=initializer),
            b=ParamInit(shape=(output_dim,), initializer=InitializerEnum.zeros),
            activation=activation,
        )


class Dropout(Module):
    """Dropout Layer. If in train mode, keeps input activations at given probability rate,
    otherwise returns inputs directly"""

    rng: Optional[Tensor] = None
    keep: float = 0.5
    mode: Mode = Mode.train

    @jit
    def forward(self, inputs: Tensor) -> Tensor:
        """If in train mode, keeps input activations at rate,
        otherwise returns directly"""

        if self.mode == Mode.eval:
            return inputs

        mask = random.bernoulli(self._rng, self.keep, inputs.shape)  # type: ignore
        return np.where(mask, inputs / self.keep, 0)

    def initialize(self, rng: Tensor) -> Dropout:
        return self.update(**{"is_initialized": True, "rng": rng})


class Sequential(Module):
    __root__: Tuple[Module, ...]

    def __len__(self) -> int:
        return len(self.__root__)

    def __iter__(self) -> Iterator[Module]:  # type: ignore
        for mod in self.__root__:
            yield mod

    def add(self, module: Module) -> Sequential:
        d = self.__root__
        new_tup = list(d) + [module]
        return Sequential(__root__=tuple(new_tup))

    @classmethod
    def build(cls, *args: Module) -> Sequential:
        return Sequential(__root__=args)

    def new_state(self, rng: Tensor, mode: str = "train") -> Sequential:
        d: Dict[str, Any] = {"mode": mode}
        rngs = random.split(rng, len(self.__root__))

        new_tup = tuple(
            mod.new_state(rng, mode) for rng, mod in zip(rngs, self.__root__)
        )
        d["__root__"] = new_tup
        return self.copy(update=d)

    @jit
    def forward(self, inputs: Tensor) -> Tensor:
        for module in self:
            inputs = module(inputs)
        return inputs
