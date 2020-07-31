import pytest
from typing import Dict, Any
from colin_net.train import Experiment


@pytest.fixture
def flattened_config() -> Dict[str, Any]:
    return {
        "experiment_name": "fizzbuzz",
        "model_config": {
            "input_dim": 10,
            "output_dim": 4,
            "hidden_dim": 50,
            "num_hidden": 2,
            "activation": "tanh",
            "dropout_keep": None,
            "initializer": "normal",
        },
        "random_seed": 42,
        "loss": "cross_entropy",
        "regularization": None,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "global_step": 2000,
        "log_every": 100,
        "model_config.hidden_dim": 2,
        "model_config.num_hidden": 8,
        "learing_rate": 0.0001,
        "model_config.activation": "relu",
    }


@pytest.fixture
def normal_config() -> Dict[str, Any]:
    return {
        "experiment_name": "fizzbuzz",
        "model_config": {
            "input_dim": 10,
            "output_dim": 4,
            "hidden_dim": 2,
            "num_hidden": 8,
            "activation": "relu",
            "dropout_keep": None,
            "initializer": "normal",
        },
        "random_seed": 42,
        "loss": "cross_entropy",
        "regularization": None,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "global_step": 2000,
        "log_every": 100,
        "learing_rate": 0.0001,
    }


def test_unflattening(
    normal_config: Dict[str, Any], flattened_config: Dict[str, Any]
) -> None:
    normal_exp = Experiment(**normal_config)
    unflattened_exp = Experiment.from_flattened(flattened_config)
    assert normal_exp.dict() == unflattened_exp.dict()

