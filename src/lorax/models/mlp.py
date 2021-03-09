"""
"""
from typing import Any, Dict, List

from lorax import nn
from lorax.nn.functional import ActivationEnum, InitializerEnum
from lorax.nn.layers import Dropout, Linear


class MLP(nn.Sequential):
    """Class for deep feed forward models like Multilayer Perceptrons."""

    @classmethod
    def create(
        cls,
        *,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: ActivationEnum = ActivationEnum.tanh,
        dropout_keep: float = None,
        initializer: InitializerEnum = InitializerEnum.normal,
        **kwargs: Dict[str, Any]
    ) -> "MLP":

        sizes = [input_dim] + hidden_sizes
        layers: List[nn.Module] = []

        for in_dim, hidden_dim in zip(sizes, sizes[1:]):
            layers.append(
                Linear.build(
                    input_dim=in_dim,
                    output_dim=hidden_dim,
                    activation=activation,
                    initializer=initializer,
                )
            )
            if dropout_keep:
                layers.append(Dropout(keep=dropout_keep))
        layers.append(
            Linear.build(
                input_dim=sizes[-1],
                output_dim=output_dim,
                activation=ActivationEnum.identity,
                initializer=initializer,
            )
        )

        return cls(__root__=layers)


__all__ = ["MLP"]
