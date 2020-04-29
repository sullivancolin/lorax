"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself.
"""
from typing import Any, Iterable, Sequence, Tuple

from jax import jit, vmap
from jax.tree_util import register_pytree_node_class

from colin_net.layers import Layer
from colin_net.tensor import Tensor


@register_pytree_node_class
class NeuralNet(Layer):
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    @jit
    def predict(self, inputs: Tensor) -> Tensor:
        """Predict for a single instance by iterting over all the layers"""
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    @jit
    def __call__(self, inputs: Tensor) -> Tensor:
        """Batched Predictions"""

        return vmap(self.predict)(inputs)

    def __repr__(self) -> str:
        return f"<NeuralNet layers={[layer.__repr__() for layer in self.layers]}"

    def tree_flatten(self) -> Tuple[Iterable[Any], None]:
        return tuple(self.layers), None

    @classmethod
    def tree_unflatten(cls, aux: Any, params: Sequence[Layer]) -> "NeuralNet":

        return cls(params)
