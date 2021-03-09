from enum import Enum

from jax.nn import leaky_relu, log_softmax, relu, selu, sigmoid, softmax, softplus
from jax.numpy import tanh

from lorax.tensor import Tensor


def mish(x: Tensor) -> Tensor:
    return x * tanh(softplus(x))


def identity(x: Tensor) -> Tensor:
    return x


class ActivationEnum(str, Enum):
    tanh = "tanh"
    relu = "relu"
    leaky_relu = "leaky_relu"
    selu = "selu"
    sigmoid = "sigmoid"
    softmax = "softmax"
    mish = "mish"
    identity = "identity"
    log_softmax = "log_softmax"


ACTIVATIONS = {
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "selu": selu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "mish": mish,
    "identity": identity,
    "log_softmax": log_softmax,
}
