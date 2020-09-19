from enum import Enum

from jax import nn

INITIALIZERS = {
    "normal": nn.initializers.normal(stddev=1.0),
    "glorot_normal": nn.initializers.glorot_normal(),
    "lecun_normal": nn.initializers.lecun_normal(),
}


class InitializerEnum(str, Enum):
    normal = "normal"
    glorot_normal = "glorot_normal"
    lecun_normal = "lecun_normal"
