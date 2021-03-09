"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
from typing import Any, Dict, Type, TypeVar

from jax import jit, vmap

from lorax import nn
from lorax.tensor import Tensor

T = TypeVar("T", bound="LSTMClassifier")


class LSTMClassifier(nn.Module):

    embeddings: nn.Embedding
    lstm: nn.LSTM
    output_layer: nn.Linear
    output_dim: int

    @classmethod
    def build(
        cls: Type[T],
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        **kwargs: Dict[str, Any]
    ) -> T:
        embedding = nn.Embedding.build(vocab_size, embedding_dim)

        lstm = nn.LSTM.build(input_dim=embedding_dim, hidden_dim=hidden_dim)

        output_layer = nn.Linear.build(
            input_dim=hidden_dim,
            output_dim=output_dim,
            activation=nn.functional.ActivationEnum.identity,
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


class BiLSTMClassifier(nn.Module):

    embeddings: nn.Embedding
    bilstm: nn.BiLSTM
    output_layer: nn.Linear
    output_dim: int

    @classmethod
    def build(
        cls,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        **kwargs: Dict[str, Any]
    ) -> "BiLSTMClassifier":
        embedding = nn.Embedding.build(vocab_size, embedding_dim)

        bilstm = nn.BiLSTM.build(input_dim=embedding_dim, hidden_dim=hidden_dim)

        output_layer = nn.Linear.build(
            input_dim=hidden_dim * 2,
            output_dim=output_dim,
            activation=nn.functional.ActivationEnum.identity,
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
    def build(cls, vocab_size: int, hidden_dim: int) -> "LSTMLanguageModel":

        lstm = nn.LSTM.build(input_dim=hidden_dim, hidden_dim=hidden_dim)

        output_layer = nn.Linear.build(input_dim=hidden_dim, output_dim=vocab_size)

        return cls(lstm=lstm, output_layer=output_layer, output_dim=vocab_size)

    @jit
    def predict(self, sequence_ids: Tensor) -> Tensor:

        sequence_embedding = self.output_layer.w.take(sequence_ids, axis=1).T
        output_sequence = self.lstm(sequence_embedding)

        return vmap(self.output_layer)(output_sequence)

    @jit
    def __call__(self, batched_sequence_ids: Tensor) -> Tensor:
        return vmap(self.predict)(batched_sequence_ids)


__all__ = ["LSTMClassifier", "BiLSTMClassifier", "LSTMSequenceTagger"]
