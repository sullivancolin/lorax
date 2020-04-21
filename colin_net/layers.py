"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Tuple, Any, Iterable
from jax import jit
from jax.tree_util import register_pytree_node_class
import jax.numpy as np
import jax
from colin_net.tensor import Tensor


LinearTuple = Tuple[Tensor, Tensor]

LinearFlattened = Tuple[LinearTuple, Any]


class Layer:
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def tree_flatten(self) -> Tuple[Iterable[Any], str]:
        raise NotImplementedError

    @classmethod
    def tree_unflatten(cls, aux: Any, params: Iterable[Any]):
        raise NotImplementedError


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
        outputs = inputs @ w + b
        """
        return inputs @ self.w + self.b

    def tree_flatten(self) -> LinearFlattened:
        return ((self.w, self.b), "Linear")

    @classmethod
    def tree_unflatten(cls, aux: Any, params: LinearTuple) -> "Linear":
        return cls(*params)


@register_pytree_node_class
class Tanh(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:
        return np.tanh(inputs)

    def tree_flatten(self) -> Tuple[Tuple[None], str]:
        return ((None,), "Tanh")

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "Tanh":
        return cls()


@register_pytree_node_class
class Relu(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:

        return jax.nn.relu(inputs)

    def tree_flatten(self) -> Tuple[Tuple[None], str]:
        return ((None,), "Relu")

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "Relu":
        return cls()


@register_pytree_node_class
class LeakyRelu(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:

        return jax.nn.leaky_relu(inputs)

    def tree_flatten(self) -> Tuple[Tuple[None], str]:
        return ((None,), "LeakyRelu")

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "LeakyRelu":
        return cls()


@register_pytree_node_class
class Sigmoid(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:

        return jax.nn.sigmoid(inputs)

    def tree_flatten(self) -> Tuple[Tuple[None], str]:
        return ((None,), "Sigmoid")

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "Sigmoid":
        return cls()


@register_pytree_node_class
class Softmax(Layer):
    def __init__(self):
        pass

    def __call__(self, inputs: Tensor) -> Tensor:

        return jax.nn.softmax(inputs)

    def tree_flatten(self) -> Tuple[Tuple[None], str]:
        return ((None,), "Softmax")

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "Softmax":
        return cls()


layer_lookup = {
    "Linear": Linear,
    "Tanh": Tanh,
    "Relu": Relu,
    "LeakyRelu": LeakyRelu,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
}
