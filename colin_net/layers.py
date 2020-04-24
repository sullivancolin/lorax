"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Any, Iterable, List, Tuple

import jax.numpy as np
from jax import jit, random, nn
from jax.random import PRNGKey
from jax.tree_util import register_pytree_node_class

from colin_net.tensor import Tensor

LinearTuple = Tuple[Tensor, Tensor]

LinearFlattened = Tuple[LinearTuple, Any]


class Layer:
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def tree_flatten(self) -> Tuple[Iterable[Any], Any]:
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux: Any, params: Iterable[Any]):
        raise NotImplementedError


class ActivationLayer(Layer):
    def tree_flatten(self) -> Tuple[List[None], None]:
        return ([None], None)

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "ActivationLayer":
        return cls()


@register_pytree_node_class
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
    def __call__(self, inputs: Tensor) -> Tensor:
        """
        outputs = w @ inputs + b
        """
        # print(inputs.shape, self.w)
        return np.dot(self.w, inputs) + self.b

    @classmethod
    def initialize(cls, *, input_size: int, output_size: int, key: PRNGKey) -> "Linear":
        return cls(
            w=random.normal(key, shape=(output_size, input_size)),
            b=random.normal(key, shape=(output_size,)),
        )

    def tree_flatten(self) -> LinearFlattened:
        return ((self.w, self.b), None)

    @classmethod
    def tree_unflatten(cls, aux: Any, params: LinearTuple) -> "Linear":
        return cls(*params)


@register_pytree_node_class
class Tanh(ActivationLayer):
    def __call__(self, inputs: Tensor) -> Tensor:
        return np.tanh(inputs)


@register_pytree_node_class
class Relu(ActivationLayer):
    def __call__(self, inputs: Tensor) -> Tensor:

        return nn.relu(inputs)


@register_pytree_node_class
class LeakyRelu(ActivationLayer):
    def __call__(self, inputs: Tensor) -> Tensor:

        return nn.leaky_relu(inputs)


@register_pytree_node_class
class Sigmoid(ActivationLayer):
    def __call__(self, inputs: Tensor) -> Tensor:

        return nn.sigmoid(inputs)


@register_pytree_node_class
class Softmax(ActivationLayer):
    def __call__(self, inputs: Tensor) -> Tensor:

        return nn.softmax(inputs)
