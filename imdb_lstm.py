from typing import List, Iterator, Tuple

import sentencepiece as spm
from tqdm.autonotebook import tqdm

from jax import random, disable_jit
import jax.numpy as np
import numpy as onp
import gzip

from colin_net.data import PaddedIterator
from colin_net.tensor import Tensor
from colin_net.nn import LSTMClassifier

tokenizer = spm.SentencePieceProcessor(model_file="imdb_tokenizer.model")


def decode(doc: List[int]) -> str:
    string_doc = tokenizer.decode([idno - 1 for idno in doc])
    return string_doc


class LabeledCorpus:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        with gzip.open(self.filename) as infile:
            self.len = sum(1 for line in infile)

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:

        with gzip.open(self.filename) as infile:
            for line in tqdm(infile, total=self.len):
                label, doc = line.decode("utf-8").strip().split("\t")
                if label == "1":
                    label_tensor = onp.array([1, 0])
                else:
                    label_tensor = onp.array([0, 1])

                doc_tensor = [int(idno) for idno in doc.split()]

                yield label_tensor, doc_tensor


train_file = "imdb_train.txt.gz"
test_file = "imdb_test.txt.gz"

train_corpus = LabeledCorpus(train_file)
test_corpus = LabeledCorpus(test_file)

train_inputs = []
train_targets = []

for label, doc in train_corpus:
    train_targets.append(label)
    train_inputs.append(doc)


key = random.PRNGKey(42)

train_iterator = PaddedIterator(train_inputs, train_targets, key=key)


test_inputs = []
test_targets = []
for label, doc in test_corpus:
    test_targets.append(label)
    test_inputs.append(doc)


test_iterator = PaddedIterator(test_inputs, test_targets, key=key)


lstm = LSTMClassifier.initialize(
    vocab_size=20001, hidden_dim=200, output_dim=2, key=key
)

print(lstm)


batch = next(train_iterator.__iter__())
outputs = lstm(batch.inputs)

print(outputs)
