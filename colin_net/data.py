"""
We'll feed inputs into our network in batches.
So here are some tools for iterating over data in batches.
"""
from typing import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.random as npr

from colin_net.tensor import Tensor


@dataclass
class Batch:

    inputs: Tensor
    targets: Tensor


class DataIterator:
    def __iter__(self) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(
        self,
        inputs: Tensor,
        targets: Tensor,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Batch]:
        starts = np.arange(0, len(self.inputs), self.batch_size)
        if self.shuffle:
            npr.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(batch_inputs, batch_targets)
