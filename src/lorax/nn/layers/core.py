"""
"""
from __future__ import annotations
from enum import Enum
from typing import Iterator, Tuple, Optional, Set

import jax.numpy as np
from jax import jit, random

from lorax.module import Module
from lorax.nn.activations import ACTIVATIONS, ActivationEnum
from lorax.nn.initilizers import INITIALIZERS, InitializerEnum
from lorax.parameter import Parameter
from lorax.rng import RNG
from lorax.tensor import Tensor


class Mode(str, Enum):
    """Allowed values for Layers with different behaviors during training and inference."""

    train = "train"
    eval = "eval"


class Linear(Module):
    """Dense Linear Layer.
    Computes output = activation(np.dot(w, inputs) + b)"""

    w: Parameter
    b: Parameter
    activation: ActivationEnum = ActivationEnum.identity

    @jit
    def forward(self, inputs: Tensor) -> Tensor:
        """outputs = self.activation(inputs * self.w + self.b)"""

        return ACTIVATIONS[self.activation](inputs @ self.w.value + self.b.value)

    @classmethod
    def initialize(
        cls,
        input_dim: int,
        output_dim: int,
        rng: RNG,
        activation: ActivationEnum = ActivationEnum.identity,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> Linear:
        """Factory for new Linear from input and output dimensions"""
        w = Parameter.from_tensor(
            INITIALIZERS[initializer](rng.to_prng(), shape=(input_dim, output_dim))
        )
        return cls(
            w=w,
            b=Parameter.from_tensor(np.zeros(shape=(output_dim,))),
            activation=activation,
        )


class Dropout(Module):
    """Dropout Layer. If in train mode, keeps input activations at given probability rate,
    otherwise returns inputs directly"""

    rng: RNG
    keep: float = 0.5
    mode: Mode = Mode.train

    @jit
    def forward(self, inputs: Tensor) -> Tensor:
        """If in train mode, keeps input activations at rate,
        otherwise returns directly"""

        if self.mode == Mode.eval:
            return inputs
        rng_key = self.rng.to_prng()
        mask = random.bernoulli(rng_key, self.keep, inputs.shape)  # type: ignore
        return np.where(mask, inputs / self.keep, 0)

    def to_eval(self) -> Dropout:
        return Dropout(rng=self.rng, keep=self.keep, mode=Mode.eval)

    def to_train(self) -> Dropout:
        return Dropout(rng=self.rng.split(num=1)[0], keep=self.keep, mode=Mode.train)


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

    def named_modules(
        self, memo: Optional[Set["Module"]] = None, prefix: str = ""
    ) -> Iterator[Tuple[str, "Module"]]:

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for index, module in enumerate(self.__root__):
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + str(index)
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def forward(self, inputs: Tensor) -> Tensor:
        for module in self:
            inputs = module(inputs)
        return inputs
