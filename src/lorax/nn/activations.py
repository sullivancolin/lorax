from enum import Enum

import jax.numpy as np
from jax import nn


class ActivationEnum(str, Enum):
    tanh = "tanh"
    relu = "relu"
    leaky_relu = "leaky_relu"
    selu = "selu"
    sigmoid = "sigmoid"
    softmax = "softmax"
    mish = "mish"
    identity = "identity"


ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "selu": nn.selu,
    "sigmoid": nn.sigmoid,
    "softmax": nn.softmax,
    "mish": lambda x: x * np.tanh(nn.softplus(x)),
    "identity": lambda x: x,
}
