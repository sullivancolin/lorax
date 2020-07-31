import json
from collections import Counter
from typing import Any, Dict, Iterator, Optional, Sequence, Type, Union

import jax.numpy as np
import wandb
from jax import random
from jax.tree_util import tree_flatten
from pydantic import BaseModel

from colin_net.config import LSTMConfig, MLPConfig
from colin_net.data import ITERATORS, DataIterator, IteratorEnum
from colin_net.loss import (
    LOSS_FUNCTIONS,
    REGULARIZATIONS,
    Loss,
    LossEnum,
    RegularizationEnum,
)
from colin_net.metrics import accuracy
from colin_net.nn import Model
from colin_net.optim import OPTIMIZERS, Optimizer, OptimizerEnum
from colin_net.tensor import Tensor


def wandb_log(d: Dict[str, Any], step: int) -> None:
    try:
        wandb.log(d, step=step)
    except Exception:
        pass


def wandb_notes(content: str) -> None:
    try:
        wandb.run.notes = content
        wandb.run.save()
    except Exception:
        pass


def wandb_save(filename: str) -> None:
    try:
        wandb.save(filename)
    except Exception:
        pass


class UpdateState(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    step: int
    loss: float
    model: Model


class Experiment(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    experiment_name: str
    model_config: Union[MLPConfig, LSTMConfig]
    random_seed: int = 42
    loss: LossEnum = LossEnum.mean_squared_error
    regularization: Optional[RegularizationEnum] = None
    optimizer: OptimizerEnum = OptimizerEnum.sgd
    learning_rate: float = 0.01
    batch_size: int = 32
    global_step: int = 5000
    log_every: int = 100

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        return cls(**d)

    @classmethod
    def from_file(cls, filename: str) -> "Experiment":
        with open(filename) as infile:
            d = json.load(infile)
            return cls.from_dict(d)

    @classmethod
    def from_flattened(cls, config: Dict[str, Any]) -> "Experiment":
        nested_fields = [field for field in config.keys() if "." in field]

        for field in nested_fields:
            new_field, inner_field = field.split(".", 1)
            if new_field not in config.keys():
                config[new_field] = {}
            config[new_field].update({inner_field: config[field]})
            del config[field]

        return cls(**config)

    def get_rng_keys(self, num: int = 1) -> Tensor:
        # breakpoint()
        if getattr(self, "key", None) is None:
            self.key: Tensor = random.PRNGKey(self.random_seed)
        keys = random.split(self.key, num=num + 1)
        if num > 1:
            subkeys = keys[1:]
        else:
            subkeys = keys[1]
        self.key = keys[0]
        return subkeys

    def create_model(self) -> Model:
        subkey = self.get_rng_keys()
        return self.model_config.initialize(subkey)

    def create_loss_func(self) -> Loss:
        if self.regularization:
            loss = REGULARIZATIONS[self.regularization](LOSS_FUNCTIONS[self.loss])
        else:
            loss = LOSS_FUNCTIONS[self.loss]
        return loss

    def create_optimizer(self, model: Model, loss: Loss) -> Optimizer:
        optimizer_class: Type[Optimizer] = OPTIMIZERS[self.optimizer]
        return optimizer_class.initialize(model, loss, self.learning_rate)

    def create_iterator(
        self, iterator_type: str, inputs: Any, targets: Any, batch_size: int
    ) -> DataIterator:
        subkey = self.get_rng_keys()
        return ITERATORS[iterator_type](
            inputs=inputs, targets=targets, key=subkey, batch_size=self.batch_size
        )

    def train(
        self,
        train_X: Sequence[Any],
        train_Y: Sequence[Any],
        test_X: Sequence[Any],
        test_Y: Sequence[Any],
        iterator_type: str = IteratorEnum.batch_iterator,
    ) -> Iterator[UpdateState]:

        model = self.create_model()

        loss = self.create_loss_func()

        optimizer = self.create_optimizer(model, loss)

        # Instantiate the data Iterators
        train_iterator = self.create_iterator(
            iterator_type, train_X, train_Y, self.batch_size
        )
        test_iterator = self.create_iterator(
            iterator_type, test_X, test_Y, self.batch_size
        )

        step = 1
        train_loss_accumulator = 0.0

        prev_accuracy = 0
        while step < self.global_step:
            for batch in train_iterator:
                model = model.to_train()
                batch_loss, model = optimizer.step(batch.inputs, batch.targets)
                grads = optimizer.grads
                params, _ = tree_flatten(grads)
                layer_names = model.get_names()
                if len(layer_names) != len(set(layer_names)):

                    counts = Counter(layer_names)
                    incrementer: Counter = Counter()
                    for weights, name in zip(params, layer_names):

                        if counts[name] > 0:
                            incrementer[name] += 1
                            counts[name] -= 1
                            new_name = f"{name}_{incrementer[name]}"
                        wandb_log(
                            {f"grad/{new_name}": wandb.Histogram(weights)}, step=step,
                        )
                else:
                    for weights, name in zip(params, layer_names):
                        wandb_log(
                            {f"grad/{name}": wandb.Histogram(weights)}, step=step,
                        )

                train_loss_accumulator += batch_loss
                if step % self.log_every == 0:
                    wandb_log(
                        {"train_loss": float(train_loss_accumulator / self.log_every)},
                        step=step,
                    )
                    train_loss_accumulator = 0.0

                    model = model.to_eval()
                    test_loss_accumulator = 0.0
                    test_pred_accumulator = []
                    test_label_accumulator = []
                    for test_batch in test_iterator:
                        test_loss = loss(model, test_batch.inputs, test_batch.targets)
                        test_label_accumulator.append(test_batch.targets)
                        test_loss_accumulator += test_loss
                        test_pred_accumulator.append(
                            model.predict_proba(test_batch.inputs)
                        )

                    test_probs = np.vstack(test_pred_accumulator)
                    test_labels = np.vstack(test_label_accumulator)
                    test_accuracy = accuracy(test_labels, test_probs)
                    if test_accuracy > prev_accuracy:
                        prev_accuracy = test_accuracy
                        model.save(
                            f"{self.experiment_name}_{step}_model.pkl", overwrite=True
                        )

                        wandb_save(f"{self.experiment_name}_{step}_model.pkl")
                    wandb_log(
                        {
                            "test_loss": float(test_loss_accumulator)
                            / test_iterator.num_batches
                        },
                        step=step,
                    )
                    wandb_log({"test_accuracy": float(test_accuracy)}, step=step)
                yield UpdateState(step=step, loss=batch_loss, model=model)
                step += 1
