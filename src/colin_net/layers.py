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


class Mode(str, Enum):
    """Allowed values for Dropout Mode"""

    train = "train"
    eval = "eval"


INITIALIZERS = {
    "normal": nn.initializers.normal(stddev=1.0),
    "glorot_normal": nn.initializers.glorot_normal(),
    "lecun_normal": nn.initializers.lecun_normal(),
}


class Initializer(str, Enum):
    normal = "normal"
    glorot_normal = "glorot_normal"
    lecun_normal = "lecun_normal"


class Layer(PyTreeLike, is_abstract=True):
    """Abstract Class for Layers. Enforces subclasses to implement
    __call__, tree_flatten, tree_unflatten and registered as Pytree"""

    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__repr__()


class ActivationLayer(Layer, is_abstract=True):
    """Abstract Class for Activation Layers."""

    def tree_flatten(self) -> Tuple[List[None], None]:
        return ([None], None)

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "ActivationLayer":
        return cls()

    def __repr__(self) -> str:
        return f"<ActivationLayer {self.__class__.__name__}>"


class Linear(Layer):
    """Dense Linear Layer.
    Computes output = np.dot(w, inputs) + b"""

    def __init__(self, w: Tensor, b: Tensor) -> None:
        self.w = w
        self.b = b

    def __repr__(self) -> str:
        return f"<LinearLayer w={self.w.shape}, b={self.b.shape}>"

    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """outputs = np.dot(w, inputs) + b in single instance notation."""

        return np.dot(self.w, inputs) + self.b

    @classmethod
    def initialize(
        cls,
        *,
        input_size: int,
        output_size: int,
        key: Tensor,
        initializer: str = Initializer.normal,
    ) -> "Linear":
        """Factory for new Linear from input and output dimentsions"""
        if initializer not in Initializer.__members__:
            raise ValueError(
                f"initializer: {initializer} not in {Initializer.__members__.values()}"
            )
        return cls(
            w=INITIALIZERS[initializer](key, shape=(output_size, input_size)),
            b=np.zeros(shape=(output_size,)),
        )

    def tree_flatten(self) -> Tuple[LinearTuple, None]:
        return ((self.w, self.b), None)

    @classmethod
    def tree_unflatten(cls, aux: Any, params: LinearTuple) -> "Linear":
        return cls(*params)


class Dropout(Layer):
    """Dropout Layer. If in train mode, keeps input activations at given probability rate,
    otherwise returns inputs directly"""

    def __init__(self, keep: float = 0.5, mode: str = Mode.train) -> None:
        self.keep = keep
        if mode not in Mode.__members__:
            raise ValueError(f"mode: {mode} not in {Mode.__members__.values()}")
        self.mode = mode

    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """If in train mode, keeps input activations at rate,
        otherwise returns directly"""

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

    def __repr__(self) -> str:
        return f"<Dropout keep={self.keep}, mode={self.mode}>"

    def tree_flatten(self) -> Tuple[List[None], Tuple[float, str]]:
        return ([None], (self.keep, self.mode))

    @classmethod
    def tree_unflatten(cls, aux: Tuple[float, str], params: List[Any]) -> "Dropout":
        return cls(*aux)


class Tanh(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return np.tanh(inputs)


class Relu(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.relu(inputs)


class LeakyRelu(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.leaky_relu(inputs)


class Selu(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.selu(inputs)


class Sigmoid(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.sigmoid(inputs)


class Softmax(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.softmax(inputs)
