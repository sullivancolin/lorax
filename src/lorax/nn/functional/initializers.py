from enum import Enum

from jax.nn.initializers import (
    he_normal,
    lecun_normal,
    normal,
    ones,
    xavier_normal,
    zeros,
)

INITIALIZERS = {
    "normal": normal(),
    "xavier_normal": xavier_normal(),
    "lecun_normal": lecun_normal(),
    "ones": ones,
    "zeros": zeros,
    "he_normal": he_normal(),
    "kaiming_normal": he_normal(),
    "glorot_normal": xavier_normal(),
}


class InitializerEnum(str, Enum):
    normal = "normal"
    xavier_normal = "xavier_normal"
    lecun_normal = "lecun_normal"
    ones = "ones"
    zeros = "zeros"
    glorot_normal = "glorot_normal"
    he_normal = "he_normal"
    kaiming_normal = "kaiming_normal"
