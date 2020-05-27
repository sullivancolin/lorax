from enum import Enum
from typing import Any, Callable, Tuple

from jax import value_and_grad
from jax.experimental.optimizers import adam
from jax.tree_util import tree_multimap

from colin_net.base import Module
from colin_net.loss import Loss
from colin_net.nn import NeuralNet
from colin_net.tensor import Tensor

LossGrad = Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[float, Tensor]]
AdamState = Tuple[Callable, Callable, Callable, Callable, Any, int]


class Optimizer(Module, is_abstract=True):
    def step(
        self, keys: Tensor, inputs: Tensor, targets: Tensor
    ) -> Tuple[float, NeuralNet]:
        raise NotImplementedError

    @classmethod
    def initialize(cls, net: NeuralNet, loss: Loss, lr: float = 0.01) -> "Optimizer":
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, net: NeuralNet, value_grad_fun: LossGrad, lr: float) -> None:
        self.net = net
        self.value_grad_fun = value_grad_fun
        self.lr = lr

    def sgd_update_combiner(self, param: Tensor, grad: Tensor) -> Tensor:
        """Convenvience method for performing SGD on custom jax Pytree objects"""
        return param - (self.lr * grad)

    def step(
        self, keys: Tensor, inputs: Tensor, targets: Tensor
    ) -> Tuple[float, NeuralNet]:
        # breakpoint()
        loss, grads = self.value_grad_fun(self.net, keys, inputs, targets)

        self.net = tree_multimap(self.sgd_update_combiner, self.net, grads)
        return loss, self.net

    @classmethod
    def initialize(cls, net: NeuralNet, loss: Loss, lr: float = 0.01) -> "SGD":
        return cls(net, value_and_grad(loss), lr)


class Adam(Optimizer):
    def __init__(
        self,
        value_grad_fun: LossGrad,
        init_fun: Callable,
        update_fun: Callable,
        get_params: Callable[..., NeuralNet],
        opt_state: Any,
        update_count: int,
    ) -> None:
        self.value_grad_fun = value_grad_fun
        self.init_fun = init_fun
        self.update_fun = update_fun
        self.get_params = get_params
        self.opt_state = opt_state
        self.update_count = update_count

    @classmethod
    def initialize(
        cls, net: NeuralNet, loss: Callable[..., Any], lr: float = 0.01
    ) -> "Adam":

        value_grad_fun = value_and_grad(loss)
        init_fun, update_fun, get_params = adam(step_size=lr)
        opt_state = init_fun(net)
        update_count = 0
        return cls(
            value_grad_fun, init_fun, update_fun, get_params, opt_state, update_count
        )

    def step(
        self, keys: Tensor, inputs: Tensor, targets: Tensor
    ) -> Tuple[float, NeuralNet]:
        net = self.get_params(self.opt_state)
        loss, grads = self.value_grad_fun(net, keys, inputs, targets)
        self.opt_state = self.update_fun(self.update_count, grads, self.opt_state)
        self.update_count += 1
        return loss, self.get_params(self.opt_state)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


class OptimizerEnum(str, Enum):
    sgd = "sgd"
    adam = "adam"
