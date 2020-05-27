"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself.
"""
import pickle
from pathlib import Path
from typing import Any, List, Union

import jax.numpy as np
from jax import jit, lax, nn, random, vmap

from colin_net.layers import (
    INITIALIZERS,
    ActivationEnum,
    ActivationLayer,
    Dropout,
    Embedding,
    InitializerEnum,
    Layer,
    Linear,
    LSTMCell,
    Mode,
)
from colin_net.tensor import Tensor

suffix = ".pkl"

glorot_normal = INITIALIZERS["glorot_normal"]


class NeuralNet(Layer, is_abstract=True):
    """Abstract Class for NeuralNet. Enforces subclasses to implement
    __call__, tree_flatten, tree_unflatten, save, load and registered as Pytree"""

    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    @jit
    def predict_proba(self, inputs: Tensor, **kwargs) -> Tensor:
        if self.output_dim > 1:
            return nn.softmax(self.__call__(inputs, **kwargs))
        else:
            return nn.sigmoid(self.__call__(inputs, **kwargs))

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
    def load(cls, path: Union[str, Path]) -> "NeuralNet":
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data


class MLP(NeuralNet):
    """Class for feed forward nets like Multilayer Perceptrons."""

    layers: List[Layer]
    input_dim: int
    output_dim: int

    @jit
    def predict(self, single_input: Tensor, key: Tensor = None) -> Tensor:
        """Predict for a single instance by iterating over all the layers"""
        for layer in self.layers:
            single_input = layer(single_input, key=key)
        return single_input

    @jit
    def __call__(self, batched_inputs: Tensor, batched_keys: Tensor = None) -> Tensor:
        """Batched Predictions"""

        return vmap(self.predict)(batched_inputs, batched_keys)

    @classmethod
    def create_mlp(
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
                input_size=input_dim,
                output_size=hidden_dim,
                key=subkey,
                initializer=initializer,
            ),
            ActivationLayer.initialize(activation),
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
            layers.append(ActivationLayer.initialize(activation))
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
        return cls(layers=layers, input_dim=input_dim, output_dim=output_dim)

    def eval(self) -> None:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                newlayer = Dropout(keep=layer.keep, mode=Mode.eval)
                self.layers[i] = newlayer

    def train(self) -> None:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                newlayer = Dropout(keep=layer.keep, mode=Mode.train)
                self.layers[i] = newlayer


class LSTMClassifier(NeuralNet):

    embeddings: Embedding
    cell: LSTMCell
    output_layer: Linear
    h_prev = Tensor
    c_prev = Tensor
    output_dim: int

    @classmethod
    def initialize(
        cls, *, vocab_size: int, hidden_dim: int, output_dim: int, key: Tensor
    ) -> "LSTMClassifier":
        key, subkey = random.split(key)
        embedding = Embedding.initialize(vocab_size, hidden_dim, subkey)

        key, subkey = random.split(key)
        cell = LSTMCell.initialize(
            input_dim=hidden_dim, hidden_dim=hidden_dim, key=subkey,
        )

        key, subkey = random.split(key)
        linear = Linear.initialize(
            input_size=hidden_dim, output_size=output_dim, key=subkey
        )

        h_prev = np.zeros(shape=(hidden_dim,))
        c_prev = np.zeros(shape=(hidden_dim,))

        return cls(
            embeddings=embedding,
            cell=cell,
            output_layer=linear,
            h_prev=h_prev,
            c_prev=c_prev,
            output_dim=output_dim,
        )

    @jit
    def predict(
        self, single_input: Tensor, h_prev: Tensor = None, c_prev: Tensor = None
    ) -> Tensor:
        if h_prev is None:
            h_prev = self.h_prev
        if c_prev is None:
            c_prev = self.c_prev

        sentence_embedding = self.embeddings(single_input)
        state_tuple, output_sequence = lax.scan(
            self.cell.__call__, init=(h_prev, c_prev), xs=sentence_embedding
        )
        # Semantics of lax.scan
        # outputs = []
        # for word_vector in sentence_embedding:
        #     (h_prev, c_prev), output = self.cell((h_prev, c_prev), word_vector)
        #     outputs.append(output)
        # return (h_new, c_new), outputs

        h_final = output_sequence[-1]
        output = self.output_layer(h_final)

        return output

    @jit
    def __call__(self, batched_inputs: Tensor, **kwargs: Any) -> Tensor:

        batched_h_prev = np.tile(self.h_prev, (batched_inputs.shape[0], 1))

        batched_c_prev = np.tile(self.c_prev, (batched_inputs.shape[0], 1))

        return vmap(self.predict)(batched_inputs, batched_h_prev, batched_c_prev)


__all__ = ["NeuralNet", "MLP", "LSTMClassifier"]
