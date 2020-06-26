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

from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor

Loss = Callable[[NeuralNet, Tensor, Tensor], float]


class LossEnum(str, Enum):
    mean_squared_error = "mean_squared_error"
    cross_entropy_loss = "binary_cross_entropy"


class RegularizationEnum(str, Enum):
    l2_reguluarized = "l2_regularized"
    l1_regularized = "l1_regularized"


@jit
def l2(net: NeuralNet) -> float:
    params, _ = tree_flatten(net)
    return np.sum([np.sum(layer ** 2) for layer in params])


def l2_reguluarized(loss: Loss) -> Callable[[NeuralNet, Tensor, Tensor], float]:
    @jit
    def wrapped(net: NeuralNet, inputs: Tensor, targets: Tensor) -> float:
        return loss(net, inputs, targets) + l2(net)

    return wrapped


@jit
def l1(net: NeuralNet) -> float:
    params, _ = tree_flatten(net)
    return np.sum([np.sum(np.abs(layer)) for layer in params])


def l1_regularized(loss: Loss) -> Callable[[NeuralNet, Tensor, Tensor], float]:
    @jit
    def wrapped(net: NeuralNet, inputs: Tensor, targets: Tensor) -> float:
        return loss(net, inputs, targets) + l1(net)

    return wrapped


@jit
def mean_squared_error(model: NeuralNet, inputs: Tensor, targets: Tensor) -> float:

    predicted = model(inputs)
    return np.mean((predicted - targets) ** 2)


@jit
def binary_cross_entropy(
    model: NeuralNet, inputs: Tensor, targets: Tensor, tol: float = 1e-10
) -> float:
    predicted = model(inputs)

    return -np.mean(
        targets * np.log(np.maximum(tol, predicted))
        + (1 - targets) * np.log(np.maximum(tol, 1 - predicted))
    )


LOSS_FUNCTIONS = {
    "mean_squared_error": mean_squared_error,
    "binary_cross_entropy": binary_cross_entropy,
}


REGULARIZATIONS = {
    "l2_regularized": l2_reguluarized,
    "l1_regularized": l1_regularized,
}
