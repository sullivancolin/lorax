"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import jax.numpy as np

from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor


def mean_sqaured_error(model: NeuralNet, inputs: Tensor, actual: Tensor) -> float:

    predicted = model(inputs)
    return np.mean((predicted - actual) ** 2)


def cross_entropy_loss(model: NeuralNet, inputs: Tensor, actual: Tensor) -> float:
    predicted = model(inputs)
    return -np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
