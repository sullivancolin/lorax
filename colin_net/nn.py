"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Tuple, Iterable, Any
from jax.tree_util import register_pytree_node_class
from colin_net.tensor import Tensor
from colin_net.layers import Layer, layer_lookup


@register_pytree_node_class
class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def __call__(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def __repr__(self) -> str:
        return f"<NeuralNet layers={[layer.__repr__() for layer in self.layers]}"

    def tree_flatten(self) -> Tuple[Iterable[Any], str]:
        return tuple(self.layers), "NeuralNet"

    @classmethod
    def tree_unflatten(cls, aux: Any, params: Iterable[Any]) -> "NeuralNet":

        layers = [
            layer_lookup[name](*val) if val[0] is not None else layer_lookup[name]()
            for val, name in params
        ]
        return cls(layers)
