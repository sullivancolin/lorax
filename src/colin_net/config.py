from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel
from typing_extensions import Literal

from colin_net.models import (
    MLP,
    BiLSTMClassifier,
    LSTMClassifier,
    LSTMSequenceTagger,
    Model,
)
from colin_net.nn.activations import ActivationEnum
from colin_net.nn.initilizers import InitializerEnum
from colin_net.rng import RNG
from colin_net.tensor import Tensor


class ModelConfig(BaseModel, ABC):
    @abstractmethod
    def initialize(self, key: Tensor) -> Model:
        raise NotImplementedError


class MLPConfig(ModelConfig):
    kind: Literal["MLP"]
    input_dim: int
    output_dim: int
    hidden_sizes: List[int]
    activation: ActivationEnum = ActivationEnum.tanh
    dropout_keep: Optional[float] = None
    initializer: InitializerEnum = InitializerEnum.normal

    def initialize(self, rng: RNG) -> MLP:
        return MLP.initialize(rng=rng, **self.dict())


class LSTMClassifierConfig(ModelConfig):
    kind: Literal["LSTMClassifier"]
    embedding_dim: int
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, rng: RNG) -> LSTMClassifier:
        return LSTMClassifier.initialize(rng=rng, **self.dict())


class BiLSTMClassifierConfig(ModelConfig):
    kind: Literal["BiLSTMClasifier"]
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, rng: RNG) -> BiLSTMClassifier:
        return BiLSTMClassifier.initialize(rng=rng, **self.dict())


class LSTMSequenceTaggerConfig(ModelConfig):
    kind: Literal["LSTMSequenceTagger"]
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, rng: RNG) -> LSTMSequenceTagger:
        return LSTMSequenceTagger.initialize(rng=rng, **self.dict())
