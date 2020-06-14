from enum import Enum
from typing import Any, Callable, Tuple

from jax import value_and_grad
from jax.experimental.optimizers import adam
from jax.tree_util import tree_multimap

from colin_net.base import Module
from colin_net.loss import Loss
from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor

LossGrad = Callable[[NeuralNet, Tensor, Tensor], Tuple[float, NeuralNet]]


class Optimizer(Module, is_abstract=True):
    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, NeuralNet]:
        raise NotImplementedError

    @classmethod
    def initialize(cls, net: NeuralNet, loss: Loss, lr: float = 0.01) -> "Optimizer":
        raise NotImplementedError


class SGD(Optimizer):

    net: NeuralNet
    value_grad_func: LossGrad
    lr: float

    def sgd_update_combiner(self, param: Tensor, grad: Tensor) -> Tensor:
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (self.lr * grad)

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, NeuralNet]:
        # breakpoint()
        loss, grads = self.value_grad_func(self.net, inputs, targets)

        self.net = tree_multimap(self.sgd_update_combiner, self.net, grads)
        return loss, self.net

    @classmethod
    def initialize(cls, net: NeuralNet, loss: Loss, lr: float = 0.01) -> "SGD":
        return cls(net=net, value_grad_func=value_and_grad(loss), lr=lr)


class Adam(Optimizer):

    value_grad_func: LossGrad
    init_func: Callable
    update_func: Callable
    get_params: Callable
    opt_state: Any
    update_count: int

    @classmethod
    def initialize(
        cls, net: NeuralNet, loss: Callable[..., Any], lr: float = 0.01
    ) -> "Adam":

        value_grad_func = value_and_grad(loss)
        init_fun, update_fun, get_params = adam(step_size=lr)
        opt_state = init_fun(net)
        update_count = 0
        return cls(
            value_grad_func=value_grad_func,
            init_fun=init_fun,
            update_fun=update_fun,
            get_params=get_params,
            opt_state=opt_state,
            update_count=update_count,
        )

    def step(self, inputs: Tensor, targets: Tensor) -> Tuple[float, NeuralNet]:
        net = self.get_params(self.opt_state)
        loss, grads = self.value_grad_func(net, inputs, targets)
        self.opt_state = self.update_func(self.update_count, grads, self.opt_state)
        self.update_count += 1
        return loss, self.get_params(self.opt_state)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"
