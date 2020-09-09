"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
import pickle
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union

import numpy as onp
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_flatten

from colin_net.nn.layers import Layer
from colin_net.tensor import Tensor

suffix = ".pkl"

T = TypeVar("T", bound="Model")


def flatten_layer_names(d: Dict[str, Any]) -> List[str]:
    keys = []
    for k, v in d.items():
        if isinstance(v, DeviceArray) or isinstance(v, onp.ndarray):
            keys.append(k)
        elif isinstance(v, dict) and "Frozen" not in k:
            keys.extend(flatten_layer_names(v))
        elif isinstance(v, list):
            keys.extend([key for ls in v for key in flatten_layer_names(ls)])
    return keys


class Model(Layer, is_abstract=True):

    """Abstract Class for a Model. Enforces subclasses to implement
    __call__, train_params, static_params"""

    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def to_train(self) -> "Model":
        return self

    def to_eval(self) -> "Model":
        return self

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        path = Path(path)
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f"File {path} already exists.")
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            instance = pickle.load(file)
        return instance

    @classmethod
    def load_state_dict(cls: Type[T], state_dict: Dict[str, Any]) -> T:
        return cls(**state_dict)

    def get_layer_names(self) -> List[str]:
        d = self.dict()
        return flatten_layer_names(d)

    def total_trainable_params(self) -> int:
        params, _ = tree_flatten(self)
        count = 0
        for tensor in params:
            count += tensor.size
        return count

    def num_trainable_params_by_layer(self) -> Dict[str, int]:
        params, _ = tree_flatten(self)
        layer_names = flatten_layer_names(self.dict())
        sizes: Dict[str, int] = {}
        if len(layer_names) != len(set(layer_names)):

            counts = Counter(layer_names)
            incrementer: Counter = Counter()

            for weights, name in zip(params, layer_names):

                if counts[name] > 0:
                    counts[name] -= 1
                    new_name = f"{name}_{incrementer[name]}"
                    incrementer[name] += 1

                    sizes[new_name] = weights.size
        else:
            for weights, name in zip(params, layer_names):
                sizes[name] = weights.size

        return sizes
