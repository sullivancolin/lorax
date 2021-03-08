from enum import Enum

from jax.nn.initializers import glorot_normal, lecun_normal, normal

INITIALIZERS = {
    "normal": normal(stddev=1.0),
    "glorot_normal": glorot_normal(),
    "lecun_normal": lecun_normal(),
}


class InitializerEnum(str, Enum):
    normal = "normal"
    glorot_normal = "glorot_normal"
    lecun_normal = "lecun_normal"
