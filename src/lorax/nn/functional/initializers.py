from enum import Enum

from jax.nn.initializers import lecun_normal, normal, ones, xavier_normal, zeros

INITIALIZERS = {
    "normal": normal(),
    "xavier_normal": xavier_normal(),
    "lecun_normal": lecun_normal(),
    "ones": ones,
    "zeros": zeros,
}


class InitializerEnum(str, Enum):
    normal = "normal"
    xavier_normal = "xavier_normal"
    lecun_normal = "lecun_normal"
    ones = "ones"
    zeros = "zeros"
