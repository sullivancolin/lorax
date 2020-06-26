"""
Here's a function that can train a neural net
"""
from typing import Iterator, Tuple

from colin_net.data import DataIterator
from colin_net.loss import Loss
from colin_net.nn import NeuralNet
from colin_net.optim import Adam


def train(
    net: NeuralNet,
    num_epochs: int,
    iterator: DataIterator,
    loss: Loss,
    lr: float = 0.01,
) -> Iterator[Tuple[int, float, NeuralNet]]:
    optimizer = Adam.initialize(net, loss, lr)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator:
            net = net.to_train()
            batch_loss, net = optimizer.step(batch.inputs, batch.targets)
            epoch_loss += batch_loss

        # Must return net as it has been reinstantiated, not mutated.
        epoch_loss = float(epoch_loss)

        yield (epoch, epoch_loss, net)
