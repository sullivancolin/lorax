from typing import Any, Dict, Tuple

import jax.numpy as np
from jax import jit, lax, nn, ops

from lorax.nn.initilizers import INITIALIZERS, InitializerEnum
from lorax.nn.layers.core import Layer
from lorax.rng import RNG
from lorax.tensor import Tensor


class Embedding(Layer):

    embedding_matrix: Tensor

    @jit
    def __call__(self, sequence_ids: Tensor) -> Tensor:
        return self.embedding_matrix[sequence_ids]

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        hidden_dim: int,
        rng: RNG,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "Embedding":
        vectors = INITIALIZERS[initializer](
            rng.to_prng(), shape=(vocab_size, hidden_dim)
        )
        vectors = ops.index_update(vectors, ops.index[0, :], 0.0)
        return cls(embedding_matrix=vectors)

    def trainable_params(self) -> Dict[str, Any]:
        return {"embedding_matrix": self.embedding_matrix}

    def static_params(self) -> Dict[str, Any]:
        return {}


class FrozenEmbedding(Embedding):
    """Untrainable Embedding Layer for pretrained embedding"""

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def static_params(self) -> Dict[str, Any]:
        return {"embedding_matrix": self.embedding_matrix}

    def trainable_params(self) -> Dict[str, Any]:
        return {}


class LSTM(Layer):

    U: Tensor
    V: Tensor
    b: Tensor

    h_prev: Tensor
    c_prev: Tensor

    @classmethod
    def initialize(cls, input_dim: int, hidden_dim: int, rng: RNG) -> "LSTM":
        U_rng, V_rng = rng.split()

        U = nn.initializers.glorot_normal()(
            U_rng.to_prng(), shape=(4 * hidden_dim, input_dim)
        )

        V = nn.initializers.glorot_normal()(
            V_rng.to_prng(), shape=(4 * hidden_dim, hidden_dim)
        )

        # Forget bias is ones instead of zeros to start
        b = np.zeros(shape=(3 * hidden_dim,))
        b_f = np.ones(shape=(hidden_dim,))

        b = np.hstack([b, b_f])

        h_prev = np.zeros(shape=(hidden_dim,))
        c_prev = np.zeros(shape=(hidden_dim,))

        return cls(U=U, V=V, b=b, h_prev=h_prev, c_prev=c_prev)

    @jit
    def time_step(
        self, state: Tuple[Tensor, Tensor], embedding: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:

        h_prev, c_prev = state

        igof = self.U @ embedding + self.V @ h_prev + self.b

        i, g, o, f = np.split(igof, 4, axis=1)

        i = nn.sigmoid(i)
        o = nn.sigmoid(o)
        f = nn.sigmoid(f)
        g = nn.tanh(g)

        c_new = f * c_prev + i * g
        h_new = o * np.tanh(c_new)

        return (h_new, c_new), h_new

    @jit
    def __call__(self, sequence_embedding: Tensor) -> Tensor:
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

    def trainable_params(self) -> Dict[str, Any]:
        return {
            "U": self.U,
            "V": self.V,
            "b": self.b,
            "c_prev": self.c_prev,
            "h_prev": self.h_prev,
        }

    def static_params(self) -> Dict[str, Any]:
        return {}


class FrozenLSTM(LSTM):
    """Untrainable LSTM pretrained layer"""

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def static_params(self) -> Dict[str, Any]:
        return {
            "U": self.U,
            "V": self.V,
            "b": self.b,
            "c_prev": self.c_prev,
            "h_prev": self.h_prev,
        }

    def trainable_params(self) -> Dict[str, Any]:
        return {}


class BiLSTM(Layer):

    forward_lstm: LSTM
    backward_lstm: LSTM

    @classmethod
    def initialize(cls, *, input_dim: int, hidden_dim: int, rng: RNG) -> "BiLSTM":

        rng_1, rng_2 = rng.split()
        forward_lstm = LSTM.initialize(
            input_dim=input_dim, hidden_dim=hidden_dim, rng=rng_1,
        )
        backward_lstm = LSTM.initialize(
            input_dim=input_dim, hidden_dim=hidden_dim, rng=rng_2
        )

        return cls(forward_lstm=forward_lstm, backward_lstm=backward_lstm,)

    @jit
    def __call__(self, sequence_embedding: Tensor) -> Tensor:

        forward_output_sequence = self.forward_lstm(sequence_embedding)

        backward_output_sequence = self.backward_lstm(
            np.flip(sequence_embedding, axis=0)
        )

        output = np.hstack((forward_output_sequence, backward_output_sequence))

        return output

    def trainable_params(self) -> Dict[str, Any]:
        return {
            "forward_lstm": self.forward_lstm,
            "backward_lstm": self.backward_lstm,
        }

    def static_params(self) -> Dict[str, Any]:
        return {}
