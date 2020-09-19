import gzip
import json
from typing import Iterator, Tuple

import numpy as onp
from tokenizers import Tokenizer
from tqdm.autonotebook import tqdm

import wandb
from lorax.tensor import Tensor
from lorax.train import Experiment, wandb_notes


class LabeledCorpus:
    def __init__(self, filename: str, max_len: int = 500) -> None:
        self.filename = filename
        with gzip.open(self.filename) as infile:
            self.len = sum(1 for line in infile)

    def __iter__(self) -> Iterator[Tuple[Tensor, str]]:

        with gzip.open(self.filename) as infile:
            for line in tqdm(infile, total=self.len, desc=f"{self.filename} Corpus"):
                label, doc = line.decode("utf-8").strip().split("\t")
                if label == "1":
                    label_tensor = onp.array([1, 0])
                else:
                    label_tensor = onp.array([0, 1])

                yield label_tensor, doc.lower()


# class PaddedIterator(DataIterator):
#     def __init__(
#         self,
#         inputs: List[str],
#         targets: List[Tensor],
#         rng: RNG,
#         max_len: int = 200,
#         batch_size: int = 32,
#     ) -> None:
#         self.inputs = inputs
#         self.targets = np.array(onp.array(targets))
#         tokenizer.enable_truncation(max_len)
#         tokenizer.enable_padding()
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.rng = rng
#         self.len = len(self.inputs)
#         self.num_batches = math.ceil(self.len / batch_size)

#     def __len__(self) -> int:
#         return self.len

#     def __iter__(self) -> Iterator[Batch]:
#         starts = np.arange(0, len(self.inputs), self.batch_size)

#         self.rng, new_rng = self.rng.split()
#         starts = random.permutation(new_rng.to_prng(), starts)
#         for start in starts:
#             end = start + self.batch_size
#             batch_inputs = self.inputs[start:end]
#             batch_targets = self.targets[start:end]

#             padded_inputs = np.array(
#                 [
#                     item.ids[-self.max_len :]
#                     if item.ids[-1] != 0
#                     else item.ids[: self.max_len]
#                     for item in self.tokenizer.encode_batch(batch_inputs)
#                 ]
#             )

#             yield Batch(padded_inputs, batch_targets)



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

tokenizer = Tokenizer.from_file("rust_tokenizer.json")
VOCAB_SIZE = len(tokenizer.get_vocab())


config = {
    "experiment_name": "imdb_lstm",
    "model_config": {
        "kind": "LSTMClassifier",
        "output_dim": 2,
        "embedding_dim": 200,
        "vocab_size": VOCAB_SIZE,
        "hidden_dim": 300,
    },
    "random_seed": 42,
    "iterator_type": "padded_iterator",
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "regularization": "l2",
    "batch_size": 32,
    "global_step": 50000,
    "log_every": 500,
}

wandb.init(project="colin_net_lstm", config=config, save_code=True)
config = wandb.config

experiment = Experiment.from_flattened(config)

print(json.dumps(experiment.dict(), indent=4))

update_generator = experiment.train(
    train_X=train_inputs,
    train_Y=train_targets,
    test_X=test_inputs,
    test_Y=test_targets,
    iterator_type="padded_iterator",
    tokenizer=tokenizer,
)

bar = tqdm(total=experiment.global_step)
for update_state in update_generator:
    if update_state.step == 1:
        markdown = f"{update_state.model.json()}"
        wandb_notes(markdown)
    if update_state.step % experiment.log_every == 0:
        bar.set_description(f"loss:{update_state.loss:.5f}")
    bar.update()


final_model = update_state.model
final_model.save("final_lstm.pkl")
