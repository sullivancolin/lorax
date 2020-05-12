"""
Here's a function that can train a neural net
"""
from typing import Iterator, Tuple

from jax import jit, random, value_and_grad
from jax.tree_util import tree_multimap

from colin_net.data import DataIterator
from colin_net.loss import Loss
from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor


def train(
    net: NeuralNet,
    key: Tensor,
    num_epochs: int,
    iterator: DataIterator,
    loss: Loss,
    lr: float = 0.01,
) -> Iterator[Tuple[int, float, NeuralNet]]:
    @jit
    def sgd_update_combiner(param, grad, lr=lr):
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (lr * grad)

    value_grad_fn = value_and_grad(loss)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator:
            num_keys = batch.inputs.shape[0]
            keys = random.split(key, num_keys + 1)
            key = keys[0]
            subkeys = keys[1:]
            batch_loss, grads = value_grad_fn(net, subkeys, batch.inputs, batch.targets)
            epoch_loss += batch_loss

            net = tree_multimap(sgd_update_combiner, net, grads)

        # Must return net other as it has been reinstantiated, not mutated.
        yield (epoch, epoch_loss, net)
