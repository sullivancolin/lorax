import gzip
from typing import Iterator, List, Tuple

import jax.numpy as np
import numpy as onp
from tqdm.autonotebook import tqdm

from colin_net.config import Experiment
from colin_net.tensor import Tensor


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
    "net_config": {"output_dim": 2, "vocab_size": VOCAB_SIZE, "hidden_dim": 200},
    "random_seed": 42,
    "iterator_type": "padded_iterator",
    "loss": "mean_squared_error",
    "optimizer": "adam",
    "learning_rate": 0.01,
    "batch_size": 64,
    "global_stel": 5000,
    "log_every": 100,
}

experiment = Experiment(**config)


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


for update_state in experiment.train(
    train_inputs, train_targets, test_inputs, test_targets
):
    if update_state.iteration % experiment.log_every == 0:
        net = update_state.net
        net = net.to_eval()
        # predicted = net.predict_proba(test_inputs)
        # acc_metric = float(accuracy(test_targets, predicted))
        # update_state.test_writer.add_scalar(
        #     "accuracy", acc_metric, update_state.iteration
        # )
        # print(f"Accuracy: {acc_metric}")
        update_state.train_writer.flush()
        update_state.test_writer.flush()
        # if acc_metric >= 0.99:
        #     print("Achieved Perfect Prediction!")
        #     break
        net = net.to_train()


final_net = update_state.net
final_net.save(f"{experiment.experiment_name}/final_model.pkl", overwrite=True)

# # Display Predictions
# final_net = final_net.to_eval()
# probabilties = final_net.predict_proba(inputs)
# for gold, prob, pred in zip(targets, probabilties, np.argmax(probabilties, axis=1)):

#     print(gold, prob, pred)

# accuracy_score = float(accuracy(targets, probabilties))
# print("Accuracy: ", accuracy_score)
