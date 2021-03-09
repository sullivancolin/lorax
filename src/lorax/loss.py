"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network.

Loss functions must take the net to be optimized as the first
argument for taking the derivative with jax.grad
"""
from enum import Enum
from typing import Callable

import jax.numpy as np
from jax import jit
from jax.tree_util import tree_flatten

import lorax.nn.functional as F
from lorax.nn import Module
from lorax.tensor import Tensor

Loss = Callable[[Module, Tensor, Tensor], float]


class LossEnum(str, Enum):
    mean_squared_error = "mean_squared_error"
    cross_entropy_loss = "cross_entropy"


class RegularizationEnum(str, Enum):
    l2 = "l2"
    l1 = "l1"
    l1_l1 = "l1_l2"


@jit
def l2(module: Module) -> float:
    params, _ = tree_flatten(module)
    return np.sum([np.sum(layer ** 2) for layer in params]) / 2


def l2_reguluarized(loss: Loss) -> Loss:
    @jit
    def wrapped(
        module: Module, inputs: Tensor, targets: Tensor, gamma: float = 0.01
    ) -> float:
        return loss(module, inputs, targets) + l2(module) * gamma

    return wrapped


@jit
def l1(model: Module) -> float:
    params, _ = tree_flatten(model)
    return np.sum([np.sum(np.abs(layer)) for layer in params])


def l1_regularized(loss: Loss) -> Loss:
    @jit
    def wrapped(
        module: Module, inputs: Tensor, targets: Tensor, gamma: float = 0.01
    ) -> float:
        return loss(module, inputs, targets) + l1(module) * gamma

    return wrapped


def l1_l2_regularized(loss: Loss) -> Loss:
    @jit
    def wrapped(
        module: Module,
        inputs: Tensor,
        targets: Tensor,
        l1_gamma: float = 0.01,
        l2_gamma: float = 0.01,
    ) -> float:
        return (
            loss(module, inputs, targets)
            + l1(module) * l1_gamma
            + l2(module) * l2_gamma
        )

    return wrapped


@jit
def mean_squared_error(module: Module, inputs: Tensor, targets: Tensor) -> float:
    predicted = module(inputs)
    return np.mean((predicted - targets) ** 2)


@jit
def cross_entropy(module: Module, inputs: Tensor, targets: Tensor) -> float:
    # log softmax
    predicted = F.log_softmax(module(inputs))

    # negative log likelihood
    return -np.mean(np.sum(targets * predicted, axis=1))


LOSS_FUNCTIONS = {
    "mean_squared_error": mean_squared_error,
    "cross_entropy": cross_entropy,
}


REGULARIZATIONS = {
    "l2": l2_reguluarized,
    "l1": l1_regularized,
    "l1_l2": l1_l2_regularized,
}
