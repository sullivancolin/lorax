"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network.

Loss functions must take the net to be optimized as the first
argument for taking the derivative with jax.grad
"""
from enum import Enum
from typing import Callable

import jax.numpy as np
from jax import jit, nn
from jax.tree_util import tree_flatten

from colin_net.models import Model
from colin_net.tensor import Tensor

Loss = Callable[[Model, Tensor, Tensor], float]


class LossEnum(str, Enum):
    mean_squared_error = "mean_squared_error"
    cross_entropy_loss = "cross_entropy"


class RegularizationEnum(str, Enum):
    l2 = "l2"
    l1 = "l1"
    l1_l1 = "l1_l2"


@jit
def l2(model: Model) -> float:
    params, _ = tree_flatten(model)
    return np.sum([np.sum(layer ** 2) for layer in params]) / 2


def l2_reguluarized(loss: Loss) -> Callable[[Model, Tensor, Tensor], float]:
    @jit
    def wrapped(
        model: Model, inputs: Tensor, targets: Tensor, gamma: float = 0.01
    ) -> float:
        return loss(model, inputs, targets) + l2(model) * gamma

    return wrapped


@jit
def l1(model: Model) -> float:
    params, _ = tree_flatten(model)
    return np.sum([np.sum(np.abs(layer)) for layer in params])


def l1_regularized(loss: Loss) -> Callable[[Model, Tensor, Tensor], float]:
    @jit
    def wrapped(
        model: Model, inputs: Tensor, targets: Tensor, gamma: float = 0.01
    ) -> float:
        return loss(model, inputs, targets) + l1(model) * gamma

    return wrapped


def l1_l2_regularized(loss: Loss) -> Callable[[Model, Tensor, Tensor], float]:
    @jit
    def wrapped(
        model: Model,
        inputs: Tensor,
        targets: Tensor,
        l1_gamma: float = 0.01,
        l2_gamma: float = 0.01,
    ) -> float:
        return (
            loss(model, inputs, targets) + l1(model) * l1_gamma + l2(model) * l2_gamma
        )

    return wrapped


@jit
def mean_squared_error(model: Model, inputs: Tensor, targets: Tensor) -> float:
    predicted = model(inputs)
    return np.mean((predicted - targets) ** 2)


@jit
def cross_entropy(model: Model, inputs: Tensor, targets: Tensor) -> float:
    # log softmax
    predicted = nn.log_softmax(model(inputs))

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
