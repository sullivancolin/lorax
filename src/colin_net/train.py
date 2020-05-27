"""
Here's a function that can train a neural net
"""
from datetime import datetime
from typing import Iterator, Tuple, Union

import jax.numpy as np
from jax import jit, random, value_and_grad
from jax.experimental.optimizers import adam
from jax.tree_util import tree_multimap
from tensorboardX import SummaryWriter

from colin_net.config import ExperimentConfig
from colin_net.data import DataIterator
from colin_net.loss import LOSS_FUNCTIONS, Loss
from colin_net.nn import MLP, NeuralNet
from colin_net.optim import OPTIMIZERS, Optimizer
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
    def sgd_update_combiner(param: Tensor, grad: Tensor, lr: float = lr) -> Tensor:
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (lr * grad)

    value_grad_fn = value_and_grad(loss)

    # init_fun, update_fun, get_params = adam(step_size=lr)
    # opt_state = init_fun(net)
    # update_count = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator:

            # batch_loss, grads = value_grad_fn(
            #     net, keys=None, inputs=batch.inputs, targets=batch.targets
            # )
            # opt_state = update_fun(update_count, grads, opt_state)
            # update_count += 1
            num_keys = batch.inputs.shape[0]
            keys = random.split(key, num_keys + 1)
            key = keys[0]
            subkeys = keys[1:]

            batch_loss, grads = value_grad_fn(net, subkeys, batch.inputs, batch.targets)
            epoch_loss += batch_loss

            net = tree_multimap(sgd_update_combiner, net, grads)
            # net = get_params(opt_state)
        # Must return net other as it has been reinstantiated, not mutated.
        epoch_loss = float(epoch_loss) / iterator.len

        yield (epoch, epoch_loss, net)


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
        save_every: float = None,
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
            now = datetime.now().isoformat()
            self.writer = SummaryWriter(f"{self.output_dir}/logs_{now}")
        if save_every:
            self.save_every = save_every
            self.mod_index = int(save_every * num_epochs)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Trainer":
        key = random.PRNGKey(config.random_seed)
        net = MLP.create_mlp(key=key, **config.net_config.dict())
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

    def train(self, iterator: DataIterator) -> Iterator[Tuple[int, float, NeuralNet]]:
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch in iterator:
                num_keys = batch.inputs.shape[0]
                keys = random.split(self.key, num_keys + 1)
                self.key = keys[0]
                subkeys = keys[1:]
                batch_loss, self.net = self.optimzer.step(
                    subkeys, batch.inputs, batch.targets
                )
                epoch_losses.append(float(batch_loss))

            # Must return net other as it has been reinstantiated, not mutated.
            epoch_loss = float(np.array(epoch_losses).mean())

            self.checkpoint("train_loss", epoch_loss, epoch)
            yield (epoch, epoch_loss, self.net)
        self.net.save(f"{self.output_dir}/final.pkl")

    def __enter__(self) -> "Trainer":
        """Use Trainer with Context Manager."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):  # type: ignore
        """Exit Context Manager and close logger."""
        self.writer.close()

    def checkpoint(self, name: str, value: Union[int, float], i: int) -> None:
        if self.log_metrics:
            self.writer.add_scalar(name, value, i)
        if self.save_every:
            if i % self.mod_index == 0:
                self.net.save(f"{self.output_dir}/epoch_{i}.pkl")
