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

from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor

Loss = Callable[[NeuralNet, Tensor, Tensor, Tensor], float]


class LossEnum(str, Enum):
    mean_sqaured_error = "mean_squared_error"
    cross_entropy_loss = "cross_entropy_loss"


@jit
def mean_squared_error(
    model: NeuralNet, keys: Tensor, inputs: Tensor, targets: Tensor
) -> float:

    predicted = model(inputs, batched_keys=keys)
    return np.mean((predicted - targets) ** 2)


@jit
def binary_cross_entropy_loss(
    model: NeuralNet, keys: Tensor, inputs: Tensor, targets: Tensor, tol: float = 1e-10
) -> float:
    predicted = model(inputs, batched_keys=keys)

    return -np.mean(
        targets * np.log(np.maximum(tol, predicted))
        + (1 - targets) * np.log(np.maximum(tol, 1 - predicted))
    )


LOSS_FUNCTIONS = {
    "mean_squared_error": mean_squared_error,
    "binary_cross_entropy_loss": binary_cross_entropy_loss,
}
