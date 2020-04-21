"""
We'll feed inputs into our network in batches.
So here are some tools for iterating over data in batches.
"""
from typing import Iterator
from dataclasses import dataclass

import jax.numpy as np
import numpy.random as npr

from colin_net.tensor import Tensor


@dataclass
class Batch:

    inputs: Tensor
    targets: Tensor


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            npr.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
