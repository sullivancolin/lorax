"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself.
"""
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Type, Union

from jax import jit, nn, random, vmap

from colin_net.layers import (
    ActivationLayer,
    Dropout,
    Initializer,
    Layer,
    Linear,
    Mode,
    Tanh,
)
from colin_net.tensor import Tensor

suffix = ".pkl"


class NeuralNet(Layer, is_abstract=True):
    """Abstract Class for NeuralNet. Enforces subclasses to implement
    __call__, tree_flatten, tree_unflatten and registered as Pytree"""

    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError


class FeedForwardNet(NeuralNet):
    """Class for feed forward nets like Multilayer Perceptrons."""

    def __init__(self, layers: List[Layer], output_dim: int) -> None:
        self.layers = layers
        self.output_dim = output_dim

    @jit
    def predict(self, inputs: Tensor, key: Tensor) -> Tensor:
        """Predict for a single instance by iterating over all the layers"""
        for layer in self.layers:
            inputs = layer(inputs, key=key)
        return inputs

    @jit
    def __call__(self, inputs: Tensor, keys: Tensor) -> Tensor:
        """Batched Predictions"""

        return vmap(self.predict)(inputs, keys)

    @jit
    def predict_proba(self, inputs: Tensor, keys: Tensor) -> Tensor:
        if self.output_dim > 1:
            return nn.softmax(self.__call__(inputs, keys))
        else:
            return nn.sigmoid(self.__call__(inputs, keys))

    @classmethod
    def create_mlp(
        cls,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden: int,
        key: Tensor,
        activation: Type[ActivationLayer] = Tanh,
        dropout_keep: float = None,
        initializer: str = Initializer.normal,
    ) -> "FeedForwardNet":
        key, subkey = random.split(key)
        layers: List[Layer] = [
            Linear.initialize(
                input_size=input_dim,
                output_size=hidden_dim,
                key=subkey,
                initializer=initializer,
            ),
            activation(),
        ]
        if dropout_keep:
            layers.append(Dropout(keep=dropout_keep))

        for _ in range(num_hidden - 2):
            key, subkey = random.split(key)
            layers.append(
                Linear.initialize(
                    input_size=hidden_dim,
                    output_size=hidden_dim,
                    key=subkey,
                    initializer=initializer,
                )
            )
            layers.append(activation())
            if dropout_keep:
                layers.append(Dropout(keep=dropout_keep))

        key, subkey = random.split(key)
        layers.append(
            Linear.initialize(
                input_size=hidden_dim,
                output_size=output_dim,
                key=subkey,
                initializer=initializer,
            )
        )
        return cls(layers, output_dim)

    def eval(self) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.mode = Mode.eval

    def train(self) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.mode = Mode.train

    def __repr__(self) -> str:

        layers = (
            "\n\t" + "\n\t".join([layer.__repr__() for layer in self.layers]) + "\n"
        )
        return f"<FeedForwardNet layers={layers}>"

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        path = Path(path)
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f"File {path} already exists.")
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeedForwardNet":
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    def tree_flatten(self) -> Tuple[List[Layer], int]:
        return self.layers, self.output_dim

    @classmethod
    def tree_unflatten(cls, aux: int, params: List[Layer]) -> "FeedForwardNet":

        return cls(params, aux)
