"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Tuple, Iterable, Any
from jax.tree_util import register_pytree_node_class
from colin_net.tensor import Tensor
from colin_net.layers import Layer
from jax.nn import softmax


@register_pytree_node_class
class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def __call__(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def predict(self, inputs: Tensor) -> Tensor:
        return softmax(self.__call__(inputs))

    def __repr__(self) -> str:
        return f"<NeuralNet layers={[layer.__repr__() for layer in self.layers]}"

    def tree_flatten(self) -> Tuple[Iterable[Any], None]:
        return tuple(self.layers), None

    @classmethod
    def tree_unflatten(cls, aux: Any, params: Sequence[Layer]) -> "NeuralNet":

        return cls(params)
