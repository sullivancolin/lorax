"""
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence, Tuple

import jax.numpy as np
from jax import random

from lorax.tensor import Tensor

Encoder = Callable[[Any, Any], Tuple[Tensor, Tensor]]


@dataclass
class Batch:

    inputs: Tensor
    targets: Tensor


class DataIterator:

    targets: Sequence[Any]
    inputs: Sequence[Any]
    batch_size: int
    rng: Tensor
    encoder: Encoder = lambda x, y: (x, y)

    def __iter__(self) -> Iterator[Batch]:
        starts = np.arange(0, len(self.inputs), self.batch_size)
        self.rng, new_rng = random.split(self.rng)
        starts = random.permutation(new_rng.to_prng(), starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(*self.encoder(batch_inputs, batch_targets))

    def __len__(self) -> int:
        return len(self.inputs)

    @property
    def num_batches(self) -> int:
        return math.ceil(len(self.inputs) / self.batch_size)
