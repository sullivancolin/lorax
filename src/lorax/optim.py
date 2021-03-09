from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar

from jax import jit, value_and_grad
from jax.experimental.optimizers import adam
from jax.tree_util import tree_multimap
from pydantic import BaseModel

from lorax.loss import Loss
from lorax.nn import Module
from lorax.tensor import Tensor

LossGrad = Callable[[Module, Tensor, Tensor], Tuple[float, Module]]

T = TypeVar("T", bound="Optimizer")


class Optimizer(BaseModel, ABC):
    loss: Loss
    model: Module
    grads: Optional[Module] = None
    value_grad_func: LossGrad
    learning_rate: float = 0.001

    @abstractmethod
    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Module]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initialize(cls: Type[T], net: Module, loss: Loss, lr: float = 0.01) -> T:
        raise NotImplementedError


@jit
def sgd_update_combiner(param: Tensor, grad: Tensor, lr: float) -> Tensor:
    """Convenvience method for performing SGD on custom jax Pytree objects"""
    return param - (lr * grad)


class SGD(Optimizer):
    @classmethod
    def initialize(
        cls, model: Module, loss: Loss, learning_rate: float = 0.01
    ) -> "SGD":
        value_grad_func = jit(value_and_grad(loss))

        return cls(
            model=model,
            value_grad_func=value_grad_func,
            learning_rate=learning_rate,
            loss=loss,
        )

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Module]:

        loss, self.grads = self.value_grad_func(self.model, inputs, targets)

        combiner = partial(sgd_update_combiner, lr=self.learning_rate)
        self.module = tree_multimap(combiner, self.model, self.grads)
        return loss, self.module


class Adam(Optimizer):
    init_func: Callable
    update_func: Callable
    get_params: Callable
    opt_state: Any
    update_count: int = 0

    @classmethod
    def initialize(
        cls, model: Module, loss: Loss, learning_rate: float = 0.01
    ) -> "Adam":

        value_grad_func: LossGrad = jit(value_and_grad(loss))
        init_fun, update_fun, get_params = adam(step_size=learning_rate)
        opt_state = init_fun(model)
        return cls(
            model=model,
            value_grad_func=value_grad_func,
            init_func=init_fun,
            update_func=update_fun,
            get_params=get_params,
            opt_state=opt_state,
            learning_rate=learning_rate,
            loss=loss,
        )

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, Module]:
        self.module = self.get_params(self.opt_state)
        loss, self.grads = self.value_grad_func(self.module, inputs, targets)
        self.opt_state = self.update_func(self.update_count, self.grads, self.opt_state)
        self.update_count += 1
        return loss, self.get_params(self.opt_state)


OPTIMIZERS: Dict[str, Type[Optimizer]] = {"sgd": SGD, "adam": Adam}


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"
