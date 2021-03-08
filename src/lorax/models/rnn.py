"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
from typing import Any, Dict, Type, TypeVar

from jax import jit, vmap

from lorax import nn
from lorax.rng import RNG
from lorax.tensor import Tensor

T = TypeVar("T", bound="LSTMClassifier")


class LSTMClassifier(nn.Module):

    embeddings: nn.Embedding
    lstm: nn.LSTM
    output_layer: nn.Linear
    output_dim: int

    @classmethod
    def initialize(
        cls: Type[T],
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: RNG,
        **kwargs: Dict[str, Any]
    ) -> T:
        rng, new_rng = rng.split()
        embedding = nn.Embedding.initialize(vocab_size, embedding_dim, new_rng)

        rng, new_rng = rng.split()
        lstm = nn.LSTM.initialize(
            input_dim=embedding_dim, hidden_dim=hidden_dim, rng=new_rng
        )

        rng, new_rng = rng.split()
        output_layer = nn.Linear.initialize(
            input_dim=hidden_dim,
            output_dim=output_dim,
            activation=nn.functional.ActivationEnum.identity,
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

    def trainable_params(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "lstm": self.lstm,
            "output_layer": self.output_layer,
        }

    def static_params(self) -> Dict[str, Any]:
        return {"output_dim": self.output_dim}


class BiLSTMClassifier(nn.Module):

    embeddings: nn.Embedding
    bilstm: nn.BiLSTM
    output_layer: nn.Linear
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
        embedding = nn.Embedding.initialize(vocab_size, embedding_dim, new_rng)

        rng, new_rng = rng.split()
        bilstm = nn.BiLSTM.initialize(
            input_dim=embedding_dim, hidden_dim=hidden_dim, rng=new_rng
        )

        rng, new_rng = rng.split()
        output_layer = nn.Linear.initialize(
            input_dim=hidden_dim * 2,
            output_dim=output_dim,
            activation=nn.functional.ActivationEnum.identity,
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


class LSTMLanguageModel(nn.Module):
    """Language model with weight tying of embedding and output layer"""

    lstm: nn.LSTM
    output_layer: nn.Linear
    output_dim: int

    @classmethod
    def initialize(
        cls, vocab_size: int, hidden_dim: int, rng: RNG
    ) -> "LSTMLanguageModel":
        rng, new_rng = rng.split()

        lstm = nn.LSTM.initialize(
            input_dim=hidden_dim, hidden_dim=hidden_dim, rng=new_rng
        )

        output_layer = nn.Linear.initialize(
            input_dim=hidden_dim, output_dim=vocab_size, rng=rng
        )

        return cls(lstm=lstm, output_layer=output_layer, output_dim=vocab_size)

    @jit
    def predict(self, sequence_ids: Tensor) -> Tensor:

        sequence_embedding = self.output_layer.w.value.take(sequence_ids, axis=1).T
        output_sequence = self.lstm(sequence_embedding)

        return vmap(self.output_layer)(output_sequence)

    @jit
    def __call__(self, batched_sequence_ids: Tensor) -> Tensor:
        return vmap(self.predict)(batched_sequence_ids)


__all__ = ["LSTMClassifier", "BiLSTMClassifier", "LSTMSequenceTagger"]
