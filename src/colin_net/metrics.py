import jax.numpy as np
from jax import jit

from colin_net.tensor import Tensor


@jit
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))
