"""
"""
from typing import Any, Dict, List

from lorax import nn
from lorax.nn.layers import Dropout, Linear
from lorax.nn.functional import ActivationEnum, InitializerEnum
from lorax.rng import RNG


class MLP(nn.Sequential):
    """Class for deep feed forward models like Multilayer Perceptrons."""

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
        layers: List[nn.Module] = []

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

        return cls(__root__=layers)


__all__ = ["MLP"]
