"""
Our neural nets will be made up of layers.
Each layer needs to pass the inputs forward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from enum import Enum
from typing import Any, Iterable, List, Tuple

import jax.numpy as np
from jax import jit, nn, random

from colin_net.base import PyTreeLike
from colin_net.tensor import Tensor

LinearTuple = Tuple[Tensor, Tensor]

LinearFlattened = Tuple[LinearTuple, Any]


class Mode(str, Enum):
    train = "train"
    eval = "eval"


class Layer(PyTreeLike, is_abstract=True):
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError


class ActivationLayer(Layer, is_abstract=True):
    def tree_flatten(self) -> Tuple[List[None], None]:
        return ([None], None)

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "ActivationLayer":
        return cls()

    def __repr__(self) -> str:
        return f"<ActivationLayer {self.__class__.__name__}>"


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """

    def __init__(self, w: Tensor, b: Tensor) -> None:
        self.w = w
        self.b = b

    def __repr__(self):
        return f"<LinearLayer w={self.w.shape}, b={self.b.shape}>"

    def __str__(self):
        return self.__repr__()

    @jit
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:
        """
        outputs = np.dot(w, inputs) + b
        """
        return np.dot(self.w, inputs) + self.b

    @classmethod
    def initialize(cls, *, input_size: int, output_size: int, key: Tensor) -> "Linear":
        """Factory for new Linear from input and output dimentsions"""
        return cls(
            w=random.normal(key, shape=(output_size, input_size)),
            b=np.zeros(shape=(output_size,)),
        )

    def tree_flatten(self) -> LinearFlattened:
        return ((self.w, self.b), None)

    @classmethod
    def tree_unflatten(cls, aux: Any, params: LinearTuple) -> "Linear":
        return cls(*params)


class Dropout(Layer):
    def __init__(self, keep: float = 0.8, mode: str = Mode.eval):
        self.keep = keep
        if mode not in Mode.__members__:
            raise ValueError(f"mode: {mode} not in {Mode.__members__.values()}")
        self.mode = mode

    @jit
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:

        if self.mode == Mode.eval:
            return inputs

        key = kwargs.get("key", None)
        if key is None:
            msg = (
                "Dropout layer requires __call__ to be called with a PRNG key "
                "argument. That is, instead of `dropout(inputs)`, call "
                "it like `dropout(inputs, key)` where `key` is a "
                "jax.random.PRNGKey value."
            )
            raise ValueError(msg)
        mask = random.bernoulli(key, self.keep, inputs.shape)
        return np.where(mask, inputs / self.keep, 0)

    def __repr__(self):
        return f"<Dropout keep={self.keep}, mode={self.mode}>"

    def __str__(self):
        return self.__repr__()

    def tree_flatten(self) -> Tuple[List[None], Tuple[float, str]]:
        return ([None], (self.keep, self.mode))

    @classmethod
    def tree_unflatten(cls, aux: Tuple[float, str], params: List[Any]) -> "Dropout":
        return cls(*aux)


class Tanh(ActivationLayer):
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:
        return np.tanh(inputs)


class Relu(ActivationLayer):
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:

        return nn.relu(inputs)


class LeakyRelu(ActivationLayer):
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:

        return nn.leaky_relu(inputs)


class Sigmoid(ActivationLayer):
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:

        return nn.sigmoid(inputs)


class Softmax(ActivationLayer):
    def __call__(self, inputs: Tensor, **kwargs) -> Tensor:

        return nn.softmax(inputs)
