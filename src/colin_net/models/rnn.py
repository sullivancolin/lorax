"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
from typing import Any, Dict

from jax import jit, nn, vmap

from colin_net.models import Model
from colin_net.nn.layers import LSTM, ActivationEnum, BiLSTM, Embedding, Linear
from colin_net.rng import RNG
from colin_net.tensor import Tensor


class LSTMClassifier(Model):

    embeddings: Embedding
    lstm: LSTM
    output_layer: Linear
    output_dim: int

    @classmethod
    def initialize(
        cls,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: RNG,
        **kwargs: Dict[str, Any]
    ) -> "LSTMClassifier":
        rng, new_rng = rng.split()
        embedding = Embedding.initialize(vocab_size, embedding_dim, new_rng)

        rng, new_rng = rng.split()
        lstm = LSTM.initialize(
            input_dim=embedding_dim, hidden_dim=hidden_dim, rng=new_rng
        )

        rng, new_rng = rng.split()
        output_layer = Linear.initialize(
            input_dim=hidden_dim,
            output_dim=output_dim,
            activation=ActivationEnum.identity,
            rng=new_rng,
        )
        return cls(
            embeddings=embedding,
            lstm=lstm,
            output_layer=output_layer,
            output_dim=output_dim,
        )

    @jit
    def predict(self, sequence_ids: Tensor) -> Tensor:
        sequence_embedding = self.embeddings(sequence_ids)
        output_sequence = self.lstm(sequence_embedding)
        output = self.output_layer(output_sequence[-1])

        return output

    @jit
    def __call__(self, batched_sequence_ids: Tensor) -> Tensor:
        return vmap(self.predict)(batched_sequence_ids)

    @jit
    def predict_proba(self, inputs: Tensor) -> Tensor:
        if self.output_dim > 1:
            return nn.softmax(self.__call__(inputs))
        else:
            return nn.sigmoid(self.__call__(inputs))

    def trainable_params(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "lstm": self.lstm,
            "output_layer": self.output_layer,
        }

    def static_params(self) -> Dict[str, Any]:
        return {"output_dim": self.output_dim}


class BiLSTMClassifier(Model):

    embeddings: Embedding
    bilstm: BiLSTM
    output_layer: Linear
    output_dim: int

    @classmethod
    def initialize(
        cls,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: RNG,
        **kwargs: Dict[str, Any]
    ) -> "BiLSTMClassifier":
        rng, new_rng = rng.split()
        embedding = Embedding.initialize(vocab_size, embedding_dim, new_rng)

        rng, new_rng = rng.split()
        bilstm = BiLSTM.initialize(
            input_dim=embedding_dim, hidden_dim=hidden_dim, rng=new_rng
        )

        rng, new_rng = rng.split()
        output_layer = Linear.initialize(
            input_dim=hidden_dim * 2,
            output_dim=output_dim,
            activation=ActivationEnum.identity,
            rng=new_rng,
        )

        return cls(
            embeddings=embedding,
            bilstm=bilstm,
            output_layer=output_layer,
            output_dim=output_dim,
        )

    @jit
    def predict(self, sequence_ids: Tensor) -> Tensor:
        sequence_embedding = self.embeddings(sequence_ids)
        bilstm_output = self.bilstm(sequence_embedding)
        output = self.output_layer(bilstm_output[-1])

        return output

    @jit
    def __call__(self, batched_sequence_ids: Tensor) -> Tensor:
        return vmap(self.predict)(batched_sequence_ids)

    @jit
    def predict_proba(self, inputs: Tensor) -> Tensor:
        if self.output_dim > 1:
            return nn.softmax(self.__call__(inputs))
        else:
            return nn.sigmoid(self.__call__(inputs))

    def trainable_params(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "bilstm": self.bilstm,
            "output_layer": self.output_layer,
        }

    def static_params(self) -> Dict[str, Any]:
        return {"output_dim": self.output_dim}


class LSTMSequenceTagger(LSTMClassifier):
    @jit
    def predict(self, sequence_ids: Tensor) -> Tensor:

        sequence_embedding = self.embeddings(sequence_ids)
        output_sequence = self.lstm(sequence_embedding)

        return vmap(self.output_layer)(output_sequence)

    @jit
    def __call__(self, batched_sequence_ids: Tensor) -> Tensor:
        return vmap(self.predict)(batched_sequence_ids)


__all__ = ["LSTMClassifier", "BiLSTMClassifier", "LSTMSequenceTagger"]
