import json
from typing import Any, Dict, Optional

from pydantic import BaseModel

from colin_net.layers import ActivationEnum, InitializerEnum
from colin_net.loss import LossEnum
from colin_net.optim import OptimizerEnum


class NetConfig(BaseModel):
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_hidden: int
    activation: Optional[ActivationEnum] = ActivationEnum.tanh
    dropout_keep: Optional[float] = None
    initializer: Optional[InitializerEnum] = InitializerEnum.normal


class ExperimentConfig(BaseModel):
    random_seed: int = 42
    net_config: NetConfig
    loss: Optional[LossEnum] = LossEnum.mean_sqaured_error
    optimizer: Optional[OptimizerEnum] = OptimizerEnum.sgd
    output_dir: str
    learning_rate: float = 0.01
    batch_size: int = 32
    num_epochs: int = 5000
    log_metrics: bool = False
    save_every: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**d)

    @classmethod
    def from_file(cls, filename: str) -> "ExperimentConfig":
        with open(filename) as infile:
            d = json.load(infile)
            return cls.from_dict(d)
