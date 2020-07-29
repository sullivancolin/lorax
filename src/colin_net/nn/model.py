"""
A Model is just a collection of layers.
It behaves a lot like a layer itself.
"""
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union

from jax import jit, nn
from jax.interpreters.xla import DeviceArray
from jax.tree_util import tree_flatten

from colin_net.layers import Layer
from colin_net.tensor import Tensor

suffix = ".pkl"


def get_keys(d: Dict[str, Any]) -> List[str]:
    keys = []
    for k, v in d.items():
        if isinstance(v, DeviceArray):
            keys.append(k)
        elif isinstance(v, dict) and "Frozen" not in k:
            keys.extend(get_keys(v))
        elif isinstance(v, list):
            keys.extend([key for ls in v for key in get_keys(ls)])
    return keys


class Model(Layer, is_abstract=True):
    output_dim: int

    """Abstract Class for Model. Enforces subclasses to implement
    __call__, tree_flatten, tree_unflatten, save, load and registered as Pytree"""

    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def to_eval(self) -> "Model":
        return self

    def to_train(self) -> "Model":
        return self

    def randomize(self) -> "Model":
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
    def load(cls, path: Union[str, Path]) -> "Model":
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data

    @jit
    def predict_proba(self, inputs: Tensor) -> Tensor:
        if self.output_dim > 1:
            return nn.softmax(self.__call__(inputs))
        else:
            return nn.sigmoid(self.__call__(inputs))

    def get_names(self) -> List[str]:
        d = self.dict()
        return get_keys(d)

    def total_trainable_params(self) -> int:
        params, _ = tree_flatten(self)
        count = 0
        for tensor in params:
            count += tensor.size
        return count

    def trainable_params_by_layer(self) -> Dict[str, int]:
        params, _ = tree_flatten(self)
        layer_names = get_keys(self.dict())
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
