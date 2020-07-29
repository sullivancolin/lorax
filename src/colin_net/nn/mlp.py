"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
from typing import List, Tuple

from jax import jit, random, vmap

from colin_net.base import RNGWrapper
from colin_net.layers import ActivationEnum, Dropout, InitializerEnum, Layer, Linear
from colin_net.nn.model import Model
from colin_net.tensor import Tensor


class MLP(Model):
    """Class for feed forward models like Multilayer Perceptrons."""

    layers: List[Layer]
    input_dim: int
    output_dim: int

    @jit
    def predict(self, single_input: Tensor) -> Tensor:
        """Predict for a single instance by iterating over all the layers"""

        for layer in self.layers:
            single_input = layer(single_input)
        return single_input

    @jit
    def __call__(self, batched_inputs: Tensor) -> Tensor:
        """Batched Predictions"""

        return vmap(self.predict)(batched_inputs)

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden: int,
        key: Tensor,
        activation: ActivationEnum = ActivationEnum.tanh,
        dropout_keep: float = None,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "MLP":
        key, subkey = random.split(key)
        layers: List[Layer] = [
            Linear.initialize(
                input_dim=input_dim,
                output_dim=hidden_dim,
                key=subkey,
                activation=activation,
                initializer=initializer,
            ),
        ]
        if dropout_keep:
            key, subkey = random.split(key)
            rng = RNGWrapper.from_prng(subkey)

            layers.append(Dropout(rng=rng, keep=dropout_keep))

        for _ in range(num_hidden - 2):
            key, subkey = random.split(key)
            layers.append(
                Linear.initialize(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    key=subkey,
                    activation=activation,
                    initializer=initializer,
                )
            )
            if dropout_keep:
                key, subkey = random.split(key)
                rng = RNGWrapper.from_prng(subkey)
                layers.append(Dropout(rng=rng, keep=dropout_keep))

        key, subkey = random.split(key)
        layers.append(
            Linear.initialize(
                input_dim=hidden_dim,
                output_dim=output_dim,
                key=subkey,
                activation=ActivationEnum.identity,
                initializer=initializer,
            )
        )

        return cls(layers=layers, input_dim=input_dim, output_dim=output_dim)

    @jit
    def to_eval(self) -> "MLP":
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                new_layer = layer.to_eval()
                self.layers[i] = new_layer
        return MLP(
            layers=self.layers, input_dim=self.input_dim, output_dim=self.output_dim
        )

    @jit
    def to_train(self) -> "MLP":
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                new_layer = layer.to_train()
                self.layers[i] = new_layer
        return MLP(
            layers=self.layers, input_dim=self.input_dim, output_dim=self.output_dim
        )

    def tree_flatten(self) -> Tuple[List[Layer], Tuple[int, int]]:
        return self.layers, (self.input_dim, self.output_dim)

    @classmethod
    def tree_unflatten(cls, aux: Tuple[int, int], params: List[Layer]) -> "MLP":
        return cls(layers=params, input_dim=aux[0], output_dim=aux[1])


__all__ = ["MLP"]
