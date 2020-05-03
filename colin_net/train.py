"""
Here's a function that can train a neural net
"""
from typing import Callable, Iterator, Tuple

import jax.numpy as np
from jax import grad, random, tree_multimap
from jax.random import PRNGKey

from colin_net.data import DataIterator
from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor

Loss = Callable[[NeuralNet, PRNGKey, Tensor, Tensor], float]


def train(
    net: NeuralNet,
    key: PRNGKey,
    num_epochs: int,
    iterator: DataIterator,
    loss: Loss,
    lr: float = 0.01,
) -> Iterator[Tuple[int, float, NeuralNet]]:
    def sgd_update_combiner(param, grad, lr=lr):
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (lr * grad)

    grad_fn = grad(loss)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator:
            num_keys = batch.inputs.shape[0]
            key, *subkeys = random.split(key, num_keys + 1)
            epoch_loss += loss(net, np.array(subkeys), batch.inputs, batch.targets)

            grads = grad_fn(net, np.array(subkeys), batch.inputs, batch.targets)

            net = tree_multimap(sgd_update_combiner, net, grads)

        # Must return net other as it has been reinstantiated, not mutated.
        yield (epoch, epoch_loss, net)
