import gzip
import json
from typing import Iterator, List, Tuple

import numpy as onp
import wandb
from tqdm.autonotebook import tqdm

from colin_net.metrics import accuracy
from colin_net.tensor import Tensor
from colin_net.train import Experiment, wandb_log, wandb_save


class LabeledCorpus:
    def __init__(self, filename: str, max_len: int = 500) -> None:
        self.filename = filename
        self.max_len = max_len
        with gzip.open(self.filename) as infile:
            self.len = sum(1 for line in infile)

    def __iter__(self) -> Iterator[Tuple[Tensor, List[int]]]:

        with gzip.open(self.filename) as infile:
            for line in tqdm(infile, total=self.len, desc=f"{self.filename} Corpus"):
                label, doc = line.decode("utf-8").strip().split("\t")
                if label == "1":
                    label_tensor = onp.array([1, 0])
                else:
                    label_tensor = onp.array([0, 1])

                doc_tensor = [int(idno) for idno in doc.split()]
                doc_tensor = doc_tensor[-self.max_len :]
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

test_inputs = []
test_targets = []
for label, doc in test_corpus:
    test_targets.append(label)
    test_inputs.append(doc)


VOCAB_SIZE = 15001

with open("imdb_tokenizer.vocab") as infile:
    vocab = ["PAD"]
    for line in infile:
        vocab.append(line.strip().split()[0])

assert len(vocab) == VOCAB_SIZE


config = {
    "experiment_name": "imdb_lstm",
    "model_config": {"output_dim": 2, "vocab_size": VOCAB_SIZE, "hidden_dim": 200},
    "random_seed": 42,
    "iterator_type": "padded_iterator",
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "regularization": "l2",
    "batch_size": 32,
    "global_step": 50000,
    "log_every": 100,
}

wandb.init(project="colin_net_lstm", config=config, save_code=True)
config = wandb.config

experiment = Experiment(**config)

print(json.dumps(experiment.dict(), indent=4))

update_generator = experiment.train(
    train_X=train_inputs,
    train_Y=train_targets,
    test_X=test_inputs,
    test_Y=test_targets,
    iterator_type="padded_iterator",
)

bar = tqdm(total=experiment.global_step)
for update_state in update_generator:
    # if update_state.step == 1:
    #     markdown = f"# Model Definition\n```json\n{update_state.model.json()}\n```"
    #     wandb_notes(markdown)
    if update_state.step % experiment.log_every == 0:
        bar.set_description(f"loss:{update_state.loss:.5f}")
    bar.update()


final_model = update_state.model
