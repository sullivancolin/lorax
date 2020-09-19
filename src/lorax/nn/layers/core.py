"""
"""
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict

import jax.numpy as np
from jax import jit, random

from lorax.module import Module
from lorax.nn.activations import ACTIVATIONS, ActivationEnum
from lorax.nn.initilizers import INITIALIZERS, InitializerEnum
from lorax.rng import RNG
from lorax.tensor import Tensor


class Mode(str, Enum):
    """Allowed values for Layers with different behaviors during training and inference."""

    train = "train"
    eval = "eval"


class Layer(Module, is_abstract=True):
    """Abstract Class for Layers. Enforces subclasses to implement __call__"""

    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    """Dense Linear Layer.
    Computes output = activation(np.dot(w, inputs) + b)"""

    w: Tensor
    b: Tensor
    activation: ActivationEnum = ActivationEnum.identity

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        """outputs = activation(np.dot(w, inputs) + b) in single instance notation."""

        return ACTIVATIONS[self.activation](np.dot(self.w, inputs) + self.b)

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        rng: RNG,
        activation: ActivationEnum = ActivationEnum.identity,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "Linear":
        """Factory for new Linear from input and output dimensions"""
        return cls(
            w=INITIALIZERS[initializer](rng.to_prng(), shape=(output_dim, input_dim)),
            b=np.zeros(shape=(output_dim,)),
            activation=activation,
        )

    def trainable_params(self) -> Dict[str, Any]:
        return {"w": self.w, "b": self.b}

    def static_params(self) -> Dict[str, Any]:
        return {"activation": self.activation}


class FrozenLinear(Linear):
    """Untrainable Linear Layer"""

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def trainable_params(self) -> Dict[str, Any]:
        return {}

    def static_params(self) -> Dict[str, Any]:
        return {"w": self.w, "b": self.b, "activation": self.activation}


class Dropout(Layer):
    """Dropout Layer. If in train mode, keeps input activations at given probability rate,
    otherwise returns inputs directly"""

    rng: RNG
    keep: float = 0.5
    mode: Mode = Mode.train

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        """If in train mode, keeps input activations at rate,
        otherwise returns directly"""

        if self.mode == Mode.eval:
            return inputs
        rng_key = self.rng.to_prng()
        mask = random.bernoulli(rng_key, self.keep, inputs.shape)
        return np.where(mask, inputs / self.keep, 0)

    def to_eval(self) -> "Dropout":
        return Dropout(rng=self.rng, keep=self.keep, mode=Mode.eval)

    def to_train(self) -> "Dropout":
        return Dropout(rng=self.rng.split(num=1)[0], keep=self.keep, mode=Mode.train)

    def trainable_params(self) -> Dict[str, Any]:
        return {}

    def static_params(self) -> Dict[str, Any]:
        return {"rng": self.rng, "keep": self.keep, "mode": self.mode}
