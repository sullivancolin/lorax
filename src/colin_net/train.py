"""
Here's a function that can train a neural net
"""
from typing import Iterator, Tuple, Union

from jax import jit, random, value_and_grad
import jax.numpy as np
from jax.tree_util import tree_multimap
from tensorboardX import SummaryWriter

from colin_net.data import DataIterator
from colin_net.loss import Loss, LOSS_FUNCTIONS
from colin_net.nn import NeuralNet, FeedForwardNet
from colin_net.tensor import Tensor
from colin_net.config import ExperimentConfig
from colin_net.optim import OPTIMIZERS, Optimizer


def train(
    net: NeuralNet,
    key: Tensor,
    num_epochs: int,
    iterator: DataIterator,
    loss: Loss,
    lr: float = 0.01,
) -> Iterator[Tuple[int, float, NeuralNet]]:
    @jit
    def sgd_update_combiner(param: Tensor, grad: Tensor, lr: float = lr) -> Tensor:
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (lr * grad)

    value_grad_fn = value_and_grad(loss)

    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in iterator:
            num_keys = batch.inputs.shape[0]
            keys = random.split(key, num_keys + 1)
            key = keys[0]
            subkeys = keys[1:]
            batch_loss, grads = value_grad_fn(net, subkeys, batch.inputs, batch.targets)
            epoch_losses.append(float(batch_loss))

            net = tree_multimap(sgd_update_combiner, net, grads)

        # Must return net other as it has been reinstantiated, not mutated.
        loss_arr = np.array(epoch_losses)
        yield (epoch, np.mean(loss_arr), net)


class Trainer:
    def __init__(
        self,
        random_seed: int,
        net: NeuralNet,
        loss: Loss,
        optimizer: Optimizer,
        output_dir: str,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        num_epochs: int = 5000,
        log_metrics: bool = False,
        checkpoint_every: float = None,
    ) -> None:
        self.random_seed = random_seed
        self.key = random.PRNGKey(random_seed)
        self.net = net
        self.loss = loss
        self.optimzer = optimizer
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_metrics = log_metrics
        if self.log_metrics:
            self.writer = SummaryWriter(f"{self.output_dir}/logs")
        self.checkpoint_every = checkpoint_every

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Trainer":
        key = random.PRNGKey(config.random_seed)
        net = FeedForwardNet.create_mlp(key=key, **config.net_config.dict())
        loss = LOSS_FUNCTIONS[config.loss]
        optimizer = OPTIMIZERS[config.optimizer].initialize(
            net, loss, config.learning_rate
        )

        config_dict = config.dict()
        config_dict.pop("net_config", None)
        config_dict.pop("optimizer", None)
        config_dict.pop("loss", None)
        seed = config.random_seed
        config_dict.pop("random_seed", None)
        return cls(seed, net, loss, optimizer, **config_dict)

    def dump_config(self) -> ExperimentConfig:
        raise NotImplementedError

    def train(self) -> NeuralNet:
        ...

    def log(self, name: str, value: Union[int, float], i: int) -> None:
        if self.log_metrics:
            self.writer(name, value, i)
