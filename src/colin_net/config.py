from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel

from colin_net.layers import ActivationEnum, InitializerEnum
from colin_net.nn import MLP, BiLSTMClassifier, LSTMClassifier, Model
from colin_net.tensor import Tensor


class ModelConfig(BaseModel):
    @abstractmethod
    def initialize(self, key: Tensor) -> Model:
        raise NotImplementedError


class MLPConfig(ModelConfig):
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_hidden: int
    activation: ActivationEnum = ActivationEnum.tanh
    dropout_keep: Optional[float] = None
    initializer: InitializerEnum = InitializerEnum.normal

    def initialize(self, key: Tensor) -> MLP:
        return MLP.initialize(key=key, **self.dict())


class LSTMConfig(ModelConfig):
    hidden_dim: int
    output_dim: int
    vocab_size: int

    def initialize(self, key: Tensor) -> LSTMClassifier:
        return LSTMClassifier.initialize(key=key, **self.dict())


class BiLSTMConfig(LSTMConfig):
    bidirectional: bool = True

    def initialize(self, key: Tensor) -> BiLSTMClassifier:
        return BiLSTMClassifier.initialize(key=key, **self.dict())
