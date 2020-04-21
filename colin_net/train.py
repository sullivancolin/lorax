"""
Here's a function that can train a neural net
"""
from typing import Callable
from colin_net.tensor import Tensor
from colin_net.nn import NeuralNet
from jax import grad, tree_multimap

# from colin_net.optim import Optimizer, SGD
from colin_net.data import DataIterator


Loss = Callable[[NeuralNet, Tensor, Tensor], float]


def update_combiner(param, grad, lr=0.002):
    return param - lr * grad


def train(
    net: NeuralNet,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int,
    iterator: DataIterator,
    loss: Loss,
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):

            epoch_loss += loss(net, inputs, batch.targets)
            grad_fn = grad(loss)

            grads = grad_fn(net, inputs, targets)

            net = tree_multimap(update_combiner, net, grads)
        print(epoch, epoch_loss)
