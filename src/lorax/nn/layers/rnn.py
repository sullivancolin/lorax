from __future__ import annotations

from typing import Tuple

import jax.numpy as np
from jax import jit, lax, nn, ops, vmap

from lorax.nn import Module
from lorax.nn.functional import InitializerEnum
from lorax.parameter import Parameter, ParamInit
from lorax.rng import RNG
from lorax.tensor import Tensor


class Embedding(Module):

    embedding_matrix: Parameter

    @jit
    def forward(self, sequence_ids: Tensor) -> Tensor:
        return self.embedding_matrix[sequence_ids]

    @classmethod
    def build(
        cls,
        vocab_size: int,
        hidden_dim: int,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "Embedding":
        vectors = ParamInit(shape=(vocab_size, hidden_dim))
        return cls(embedding_matrix=vectors)

    def initialize(self, rng: RNG) -> Embedding:
        embedding = super().initialize(rng)
        vectors = embedding.embedding_matrix
        vectors = ops.index_update(vectors, ops.index[0, :], 0.0)
        return self.update(embedding_matrix=vectors, _rng=embedding._rng)


class LSTM(Module):

    U: Parameter
    V: Parameter
    b: Parameter

    h_prev: Parameter
    c_prev: Parameter

    @classmethod
    def build(cls, input_dim: int, hidden_dim: int) -> "LSTM":

        U = ParamInit(
            shape=(input_dim, 4 * hidden_dim), initializer=InitializerEnum.xavier_normal
        )

        V = ParamInit(
            shape=(hidden_dim, 4 * hidden_dim),
            initializer=InitializerEnum.xavier_normal,
        )

        # Forget bias is ones instead of zeros to start
        b = np.zeros(shape=(3 * hidden_dim,))
        b_f = np.ones(shape=(hidden_dim,))

        b = np.hstack([b, b_f])

        h_prev = np.zeros(shape=(hidden_dim,))
        c_prev = np.zeros(shape=(hidden_dim,))

        return cls(
            U=U,
            V=V,
            b=b,
            h_prev=h_prev,
            c_prev=c_prev,
        )

    @jit
    def time_step(
        self, state: Tuple[Tensor, Tensor], embedding: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:

        h_prev, c_prev = state

        igof = embedding @ self.U + h_prev @ self.V + self.b

        i, g, o, f = np.split(igof, 4, axis=-1)

        i = nn.sigmoid(i)
        o = nn.sigmoid(o)
        f = nn.sigmoid(f)
        g = np.tanh(g)

        c_new = f * c_prev + i * g
        h_new = o * np.tanh(c_new)

        return (h_new, c_new), h_new

    @jit
    def single_forward(self, sequence_embedding: Tensor) -> Tensor:
        h_prev = self.h_prev
        c_prev = self.c_prev

        @jit
        def wrapped_time_step(
            state: Tuple[Tensor, Tensor], inputs: Tensor
        ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
            return self.time_step(state, inputs)

        state_tuple, output_sequence = lax.scan(
            wrapped_time_step, init=(h_prev, c_prev), xs=sequence_embedding
        )
        # Semantics of lax.scan
        # outputs = []
        # for time_step_vector in sequence_embedding:
        #     (h_prev, c_prev), output = self.lstm((h_prev, c_prev), time_step_vector)
        #     outputs.append(output)
        # return (h_new, c_new), outputs

        return output_sequence

    def forward(self, inputs: Tensor) -> Tensor:
        return vmap(self.single_forward)(inputs)


class BiLSTM(Module):

    forward_lstm: LSTM
    backward_lstm: LSTM

    @classmethod
    def build(cls, *, input_dim: int, hidden_dim: int) -> "BiLSTM":
        forward_lstm = LSTM.build(input_dim, hidden_dim)
        backward_lstm = LSTM.build(input_dim, hidden_dim)

        return cls(
            forward_lstm=forward_lstm,
            backward_lstm=backward_lstm,
        )

    @jit
    def forward(self, sequence_embedding: Tensor) -> Tensor:

        forward_output_sequence = self.forward_lstm(sequence_embedding)

        backward_output_sequence = self.backward_lstm(
            np.flip(sequence_embedding, axis=0)
        )

        output = np.hstack((forward_output_sequence, backward_output_sequence))

        return output
