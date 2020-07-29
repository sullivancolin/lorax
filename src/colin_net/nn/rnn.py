"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
from typing import Tuple

import jax.numpy as np
from jax import jit, lax, random, vmap

from colin_net.layers import ActivationEnum, Embedding, Linear, LSTMCell
from colin_net.nn.model import Model
from colin_net.tensor import Tensor

LSTMTuple = Tuple[Embedding, LSTMCell, Tensor, Tensor, Linear]


class LSTMClassifier(Model):

    embeddings: Embedding
    cell: LSTMCell
    output_layer: Linear
    h_prev: Tensor
    c_prev: Tensor
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
        output_layer = Linear.initialize(
            input_dim=hidden_dim,
            output_dim=output_dim,
            activation=ActivationEnum.identity,
            key=subkey,
        )

        h_prev = np.zeros(shape=(hidden_dim,))
        c_prev = np.zeros(shape=(hidden_dim,))

        return cls(
            embeddings=embedding,
            cell=cell,
            h_prev=h_prev,
            c_prev=c_prev,
            output_layer=output_layer,
            output_dim=output_dim,
        )

    @jit
    def predict(self, single_input: Tensor) -> Tensor:
        h_prev = self.h_prev
        c_prev = self.c_prev

        @jit
        def wrapped_cell(
            state: Tuple[Tensor, Tensor], inputs: Tensor
        ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
            return self.cell(state, inputs)

        sentence_embedding = self.embeddings(single_input)
        state_tuple, output_sequence = lax.scan(
            wrapped_cell, init=(h_prev, c_prev), xs=sentence_embedding
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
    def __call__(self, batched_inputs: Tensor) -> Tensor:
        return vmap(self.predict)(batched_inputs)

    def tree_flatten(self,) -> Tuple[LSTMTuple, int]:
        return (
            (self.embeddings, self.cell, self.h_prev, self.c_prev, self.output_layer),
            self.output_dim,
        )

    @classmethod
    def tree_unflatten(cls, aux: int, params: LSTMTuple,) -> "LSTMClassifier":
        return cls.construct(
            embeddings=params[0],
            cell=params[1],
            h_prev=params[2],
            c_prev=params[3],
            output_layer=params[4],
            output_dim=aux,
        )


BiLSTMTuple = Tuple[
    Embedding, LSTMCell, Tensor, Tensor, LSTMCell, Tensor, Tensor, Linear
]


class BiLSTMClassifier(Model):

    embeddings: Embedding
    forward_cell: LSTMCell
    forward_h_prev: Tensor
    forward_c_prev: Tensor
    backward_cell: LSTMCell
    backward_h_prev: Tensor
    backward_c_prev: Tensor
    output_layer: Linear
    output_dim: int

    @classmethod
    def initialize(
        cls, *, vocab_size: int, hidden_dim: int, output_dim: int, key: Tensor
    ) -> "BiLSTMClassifier":
        key, subkey = random.split(key)
        embedding = Embedding.initialize(vocab_size, hidden_dim, subkey)

        key, subkey = random.split(key)
        forward_cell = LSTMCell.initialize(
            input_dim=hidden_dim, hidden_dim=hidden_dim, key=subkey,
        )
        key, subkey = random.split(key)
        backward_cell = LSTMCell.initialize(
            input_dim=hidden_dim, hidden_dim=hidden_dim, key=subkey,
        )

        key, subkey = random.split(key)
        output_layer = Linear.initialize(
            input_dim=hidden_dim * 2,
            output_dim=output_dim,
            activation=ActivationEnum.identity,
            key=subkey,
        )

        forward_h_prev = np.zeros(shape=(hidden_dim,))
        forward_c_prev = np.zeros(shape=(hidden_dim,))
        backward_h_prev = np.zeros(shape=(hidden_dim,))
        backward_c_prev = np.zeros(shape=(hidden_dim,))

        return cls(
            embeddings=embedding,
            forward_cell=forward_cell,
            backward_cell=backward_cell,
            forward_h_prev=forward_h_prev,
            backward_h_prev=backward_h_prev,
            forward_c_prev=forward_c_prev,
            backward_c_prev=backward_c_prev,
            output_layer=output_layer,
            output_dim=output_dim,
        )

    @jit
    def predict(self, single_input: Tensor) -> Tensor:
        forward_h_prev = self.forward_h_prev
        forward_c_prev = self.forward_c_prev

        backward_h_prev = self.backward_h_prev
        backward_c_prev = self.backward_c_prev

        @jit
        def wrapped_forward_cell(
            state: Tuple[Tensor, Tensor], inputs: Tensor
        ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
            return self.forward_cell(state, inputs)

        @jit
        def wrapped_backward_cell(
            state: Tuple[Tensor, Tensor], inputs: Tensor
        ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
            return self.backward_cell(state, inputs)

        sentence_embedding = self.embeddings(single_input)
        forward_state_tuple, forward_output_sequence = lax.scan(
            wrapped_forward_cell,
            init=(forward_h_prev, forward_c_prev),
            xs=sentence_embedding,
        )

        backward_state_tuple, backward_output_sequence = lax.scan(
            wrapped_backward_cell,
            init=(backward_h_prev, backward_c_prev),
            xs=sentence_embedding,
            reverse=True,
        )

        h_final = np.hstack((forward_output_sequence[-1], backward_output_sequence[-1]))
        output = self.output_layer(h_final)

        return output

    @jit
    def __call__(self, batched_inputs: Tensor) -> Tensor:
        return vmap(self.predict)(batched_inputs)

    def tree_flatten(self,) -> Tuple[BiLSTMTuple, int]:
        return (
            (
                self.embeddings,
                self.forward_cell,
                self.forward_h_prev,
                self.forward_c_prev,
                self.backward_cell,
                self.backward_h_prev,
                self.backward_c_prev,
                self.output_layer,
            ),
            self.output_dim,
        )

    @classmethod
    def tree_unflatten(cls, aux: int, params: BiLSTMTuple) -> "BiLSTMClassifier":
        return cls.construct(
            embeddings=params[0],
            forward_cell=params[1],
            forward_h_prev=params[2],
            forward_c_prev=params[3],
            backward_cell=params[4],
            backward_h_prev=params[5],
            backward_c_prev=params[6],
            output_layer=params[7],
            output_dim=aux,
        )


__all__ = ["LSTMClassifier", "BiLSTMClassifier"]
