import json
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import jax.numpy as np
import wandb
from jax.tree_util import tree_flatten
from pydantic import BaseModel

from lorax.config import (
    BiLSTMClassifierConfig,
    LSTMClassifierConfig,
    LSTMSequenceTaggerConfig,
    MLPConfig,
)
from lorax.data import DataIterator
from lorax.loss import (
    LOSS_FUNCTIONS,
    REGULARIZATIONS,
    Loss,
    LossEnum,
    RegularizationEnum,
)
from lorax.metrics import accuracy
from lorax.models import Model
from lorax.optim import OPTIMIZERS, Optimizer, OptimizerEnum
from lorax.rng import RNG
from lorax.tensor import Tensor


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


def log_grads(grads: Model, step: int) -> None:
    params, _ = tree_flatten(grads)
    layer_names = grads.get_layer_names()
    if len(layer_names) != len(set(layer_names)):

        counts: Counter = Counter(layer_names)
        incrementer: Counter = Counter()
        for weights, name in zip(params, layer_names):

            if counts[name] > 0:
                incrementer[name] += 1
                counts[name] -= 1
                name = f"{name}_{incrementer[name]}"
            wandb_log(
                {f"grad/{name}": wandb.Histogram(weights)},
                step=step,
            )
    else:
        for weights, name in zip(params, layer_names):
            wandb_log(
                {f"grad/{name}": wandb.Histogram(weights)},
                step=step,
            )


def log_params(model: Model, step: int) -> None:
    params, _ = tree_flatten(model)
    layer_names = params.get_layer_names()
    if len(layer_names) != len(set(layer_names)):

        counts: Counter = Counter(layer_names)
        incrementer: Counter = Counter()
        for weights, name in zip(params, layer_names):

            if counts[name] > 0:
                incrementer[name] += 1
                counts[name] -= 1
                name = f"{name}_{incrementer[name]}"
            wandb_log(
                {f"params/{name}": wandb.Histogram(weights)},
                step=step,
            )
    else:
        for weights, name in zip(params, layer_names):
            wandb_log(
                {f"params/{name}": wandb.Histogram(weights)},
                step=step,
            )


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
    model_config: Union[
        MLPConfig,
        LSTMClassifierConfig,
        BiLSTMClassifierConfig,
        LSTMSequenceTaggerConfig,
    ]
    random_seed: int = 42
    loss: LossEnum = LossEnum.mean_squared_error
    regularization: Optional[RegularizationEnum] = None
    optimizer: OptimizerEnum = OptimizerEnum.sgd
    learning_rate: float = 0.01
    batch_size: int = 32
    global_step: int = 5000
    log_every: int = 100
    log_params: bool = False
    log_grads: bool = True
    eval: bool = False

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

    def split_rng(self, num: int = 2) -> Tuple[RNG, ...]:
        if getattr(self, "rng", None) is None:
            self.rng: RNG = RNG.from_seed(self.random_seed)
        rngs = self.rng.split(num=num)
        return rngs

    def create_model(self) -> Model:
        self.rng, new_rng = self.split_rng()
        return self.model_config.initialize(new_rng)

    def create_loss_func(self) -> Loss:
        if self.regularization:
            loss = REGULARIZATIONS[self.regularization](LOSS_FUNCTIONS[self.loss])
        else:
            loss = LOSS_FUNCTIONS[self.loss]
        return loss

    def create_optimizer(self, model: Model, loss: Loss) -> Optimizer:
        optimizer_class = OPTIMIZERS[self.optimizer]
        return optimizer_class.initialize(model, loss, self.learning_rate)

    def train(
        self, train_iterator: DataIterator, test_iterator: DataIterator
    ) -> Iterator[UpdateState]:

        model = self.create_model()

        loss = self.create_loss_func()

        optimizer = self.create_optimizer(model, loss)

        step = 1
        train_loss_accumulator = 0.0
        prev_accuracy = 0.0
        while step < self.global_step:
            for batch in train_iterator:
                model = model.to_train()
                batch_loss, model = optimizer.step(batch.inputs, batch.targets)
                grads = optimizer.grads

                train_loss_accumulator += batch_loss
                if step % 10 == 0 and grads is not None:
                    log_grads(grads, step)
                    if self.log_params:
                        log_params(model, step)

                if step % self.log_every == 0:
                    wandb_log(
                        {"train_loss": float(train_loss_accumulator / self.log_every)},
                        step=step,
                    )
                    train_loss_accumulator = 0.0

                    if self.eval:
                        self.do_eval(model, loss, test_iterator, step, prev_accuracy)
                yield UpdateState(step=step, loss=batch_loss, model=model)
                step += 1

    def do_eval(
        self,
        model: Model,
        loss: Loss,
        test_iterator: DataIterator,
        step: int,
        prev_accuracy: float,
    ) -> None:
        model = model.to_eval()
        test_loss_accumulator = 0.0
        test_pred_accumulator = []
        test_label_accumulator = []
        for test_batch in test_iterator:
            test_loss = loss(model, test_batch.inputs, test_batch.targets)
            test_label_accumulator.append(test_batch.targets)
            test_loss_accumulator += test_loss
            test_pred_accumulator.append(model.predict_proba(test_batch.inputs))

        test_probs = np.vstack(test_pred_accumulator)
        test_labels = np.vstack(test_label_accumulator)
        test_accuracy = accuracy(test_labels, test_probs)
        if test_accuracy > prev_accuracy:
            prev_accuracy = test_accuracy
            model.save(f"{self.experiment_name}_{step}_model.pkl", overwrite=True)

            wandb_save(f"{self.experiment_name}_{step}_model.pkl")
        wandb_log(
            {"test_loss": float(test_loss_accumulator) / test_iterator.num_batches},
            step=step,
        )
        wandb_log({"test_accuracy": float(test_accuracy)}, step=step)
