"""
We'll feed inputs into our network in batches.
So here are some tools for iterating over data in batches.
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, List

import jax.numpy as np
import numpy as onp
from jax import random
from tokenizers import Tokenizer

from colin_net.tensor import Tensor


@dataclass
class Batch:

    inputs: Tensor
    targets: Tensor


class DataIterator:

    len: int
    num_batches: int
    targets: Any
    inputs: Any
    batch_size: int

    def __iter__(self) -> Iterator[Batch]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(
        self, inputs: Tensor, targets: Tensor, key: Tensor, batch_size: int = 32,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.key = key
        self.len = len(self.inputs)
        self.num_batches = math.ceil(self.len / batch_size)

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> Iterator[Batch]:
        starts = np.arange(0, len(self.inputs), self.batch_size)
        self.key, subkey = random.split(self.key)
        starts = random.permutation(subkey, starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]
            yield Batch(batch_inputs, batch_targets)


class PaddedIterator(DataIterator):
    def __init__(
        self,
        inputs: List[str],
        targets: List[Tensor],
        key: Tensor,
        tokenizer: Tokenizer,
        max_len: int = 200,
        batch_size: int = 32,
    ) -> None:
        self.inputs = inputs
        self.targets = np.array(onp.array(targets))
        tokenizer.enable_truncation(max_len)
        tokenizer.enable_padding()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.key = key
        self.len = len(self.inputs)
        self.num_batches = math.ceil(self.len / batch_size)

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> Iterator[Batch]:
        starts = np.arange(0, len(self.inputs), self.batch_size)

        self.key, subkey = random.split(self.key)
        starts = random.permutation(subkey, starts)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_targets = self.targets[start:end]

            padded_inputs = np.array(
                [
                    item.ids[-self.max_len :]
                    if item.ids[-1] != 0
                    else item.ids[: self.max_len]
                    for item in self.tokenizer.encode_batch(batch_inputs)
                ]
            )

            yield Batch(padded_inputs, batch_targets)


class IteratorEnum(str, Enum):
    batch_iterator = "batch_iterator"
    padded_iterator = "padded_iterator"


ITERATORS = {"batch_iterator": BatchIterator, "padded_iterator": PaddedIterator}
