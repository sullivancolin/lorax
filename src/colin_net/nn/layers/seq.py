from typing import Any, Dict, Tuple

import jax.numpy as np
from jax import jit, lax, nn, ops

from colin_net.nn.initilizers import INITIALIZERS, InitializerEnum
from colin_net.nn.layers.core import Layer
from colin_net.rng import RNG
from colin_net.tensor import Tensor


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

    Wf: Tensor
    bf: Tensor
    Wi: Tensor
    bi: Tensor
    Wc: Tensor
    bc: Tensor
    Wo: Tensor
    bo: Tensor
    h_prev: Tensor
    c_prev: Tensor

    @classmethod
    def initialize(cls, input_dim: int, hidden_dim: int, rng: RNG) -> "LSTM":

        Wf_rng, Wi_rng, Wc_rng, Wo_rng = rng.split(num=4)

        concat_size = input_dim + hidden_dim

        Wf = nn.initializers.glorot_normal()(
            Wf_rng.to_prng(), shape=(hidden_dim, concat_size)
        )
        bf = np.zeros(shape=(hidden_dim,))

        Wi = nn.initializers.glorot_normal()(
            Wi_rng.to_prng(), shape=(hidden_dim, concat_size)
        )
        bi = np.zeros(shape=(hidden_dim,))

        Wc = nn.initializers.glorot_normal()(
            Wc_rng.to_prng(), shape=(hidden_dim, concat_size)
        )
        bc = np.zeros(shape=(hidden_dim,))

        Wo = nn.initializers.glorot_normal()(
            Wo_rng.to_prng(), shape=(hidden_dim, concat_size)
        )
        bo = np.zeros(shape=(hidden_dim,))

        h_prev = np.zeros(shape=(hidden_dim,))
        c_prev = np.zeros(shape=(hidden_dim,))

        return cls(
            Wf=Wf,
            bf=bf,
            Wi=Wi,
            bi=bi,
            Wc=Wc,
            bc=bc,
            Wo=Wo,
            bo=bo,
            c_prev=c_prev,
            h_prev=h_prev,
        )

    @jit
    def time_step(
        self, state: Tuple[Tensor, Tensor], inputs: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:

        h_prev, c_prev = state
        concat_vec = np.hstack((inputs, h_prev))

        f = nn.sigmoid(np.dot(self.Wf, concat_vec) + self.bf)
        i = nn.sigmoid(np.dot(self.Wi, concat_vec) + self.bi)
        C_bar = np.tanh(np.dot(self.Wc, concat_vec) + self.bc)

        c = f * c_prev + i * C_bar
        o = nn.sigmoid(np.dot(self.Wo, concat_vec) + self.bo)
        h = o * np.tanh(c)

        # hidden state vector is copied as output
        return (h, c), h

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
            "Wf": self.Wf,
            "bf": self.bf,
            "Wi": self.Wi,
            "bi": self.bi,
            "Wc": self.Wc,
            "bc": self.bc,
            "Wo": self.Wo,
            "bo": self.bo,
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
            "Wf": self.Wf,
            "bf": self.bf,
            "Wi": self.Wi,
            "bi": self.bi,
            "Wc": self.Wc,
            "bc": self.bc,
            "Wo": self.Wo,
            "bo": self.bo,
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
