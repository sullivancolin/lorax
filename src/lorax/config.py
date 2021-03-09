from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel
from typing_extensions import Literal

from lorax.models import MLP, BiLSTMClassifier, LSTMClassifier, LSTMSequenceTagger
from lorax.nn import Module
from lorax.nn.functional import ActivationEnum, InitializerEnum
from lorax.tensor import Tensor


class ModelConfig(BaseModel, ABC):
    @abstractmethod
    def initialize(self, key: Tensor) -> Module:
        raise NotImplementedError


class MLPConfig(ModelConfig):
    kind: Literal["MLP"]
    input_dim: int
    output_dim: int
    hidden_sizes: List[int]
    activation: ActivationEnum = ActivationEnum.tanh
    dropout_keep: Optional[float] = None
    initializer: InitializerEnum = InitializerEnum.normal

    def initialize(self, rng: Tensor) -> MLP:
        mlp = MLP.create(**self.dict())
        return mlp.initialize(rng)


class LSTMClassifierConfig(ModelConfig):
    kind: Literal["LSTMClassifier"]
    embedding_dim: int
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, rng: Tensor) -> LSTMClassifier:
        lstm = LSTMClassifier.create(**self.dict())
        return lstm.initialize(rng=rng)


class BiLSTMClassifierConfig(ModelConfig):
    kind: Literal["BiLSTMClasifier"]
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, rng: Tensor) -> BiLSTMClassifier:
        bilstm = BiLSTMClassifier.create(**self.dict())
        return bilstm.initialize(rng=rng)


class LSTMSequenceTaggerConfig(ModelConfig):
    kind: Literal["LSTMSequenceTagger"]
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, rng: Tensor) -> LSTMSequenceTagger:
        lstm_tagger = LSTMSequenceTagger.create(**self.dict())
        return lstm_tagger.initialize(rng=rng)
