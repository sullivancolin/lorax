"""
"""
from typing import Any, Dict, List

from jax import jit, vmap

from lorax.models import Model
from lorax.nn.layers import (
    ActivationEnum,
    Dropout,
    InitializerEnum,
    Layer,
    Linear,
    Mode,
)
from lorax.rng import RNG
from lorax.tensor import Tensor


class MLP(Model):
    """Class for deep feed forward models like Multilayer Perceptrons."""

    layers: List[Layer]
    input_dim: int
    output_dim: int

    @jit
    def predict(self, single_input: Tensor) -> Tensor:
        """Predict for a single instance by iterating over all the layers."""

        for layer in self.layers:
            single_input = layer(single_input)
        return single_input

    @jit
    def __call__(self, batched_inputs: Tensor) -> Tensor:
        """Batched Predictions"""

        return vmap(self.predict)(batched_inputs)

    def _set_mode(self, mode: Mode = Mode.train) -> "MLP":
        new_layers: List[Layer] = []
        for layer in self.layers:
            if isinstance(layer, Dropout):
                if mode == "train":
                    new_layers.append(layer.to_train())
                else:
                    new_layers.append(layer.to_eval())
            else:
                new_layers.append(layer)
        return MLP(
            layers=new_layers, input_dim=self.input_dim, output_dim=self.output_dim
        )

    def to_train(self) -> "MLP":
        return self._set_mode(Mode.train)

    def to_eval(self) -> "MLP":
        return self._set_mode(Mode.eval)

    @classmethod
    def initialize(
        cls,
        *,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        rng: RNG,
        activation: ActivationEnum = ActivationEnum.tanh,
        dropout_keep: float = None,
        initializer: InitializerEnum = InitializerEnum.normal,
        **kwargs: Dict[str, Any]
    ) -> "MLP":

        sizes = [input_dim] + hidden_sizes
        layers: List[Layer] = []

        for in_dim, hidden_dim in zip(sizes, sizes[1:]):
            rng, new_rng = rng.split()
            layers.append(
                Linear.initialize(
                    input_dim=in_dim,
                    output_dim=hidden_dim,
                    rng=new_rng,
                    activation=activation,
                    initializer=initializer,
                )
            )
            if dropout_keep:
                rng, new_rng = rng.split()
                layers.append(Dropout(rng=new_rng, keep=dropout_keep))
        rng, new_rng = rng.split()
        layers.append(
            Linear.initialize(
                input_dim=sizes[-1],
                output_dim=output_dim,
                rng=new_rng,
                activation=ActivationEnum.identity,
                initializer=initializer,
            )
        )

        return cls(layers=layers, input_dim=input_dim, output_dim=output_dim)

    def trainable_params(self) -> Dict[str, Any]:
        return {"layers": self.layers}

    def static_params(self) -> Dict[str, Any]:
        return {"input_dim": self.input_dim, "output_dim": self.output_dim}


__all__ = ["MLP"]
