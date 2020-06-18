import gzip
from datetime import datetime
from typing import Iterator, List, Tuple

import jax.numpy as np
import numpy as onp
import sentencepiece as spm
from jax import random, disable_jit
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from colin_net.data import PaddedIterator
from colin_net.loss import mean_squared_error
from colin_net.nn import LSTMClassifier
from colin_net.tensor import Tensor
from colin_net.train import train

tokenizer = spm.SentencePieceProcessor(model_file="imdb_tokenizer.model")


def decode(doc: List[int]) -> str:
    string_doc = tokenizer.decode([idno - 1 for idno in doc])
    return string_doc


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


key = random.PRNGKey(42)

train_iterator = PaddedIterator(train_inputs, train_targets, key=key, batch_size=64)


test_inputs = []
test_targets = []
for label, doc in test_corpus:
    test_targets.append(label)
    test_inputs.append(doc)


test_iterator = PaddedIterator(
    test_inputs[:1000], test_targets[:1000], key=key, batch_size=64
)

VOCAB_SIZE = 15001

with open("imdb_tokenizer.vocab") as infile:
    vocab = ["PAD"]
    for line in infile:
        vocab.append(line.strip().split()[0])

assert len(vocab) == VOCAB_SIZE

lstm = LSTMClassifier.initialize(
    vocab_size=VOCAB_SIZE, hidden_dim=200, output_dim=2, key=key
)

now = datetime.now().isoformat()

train_writer = SummaryWriter(f"lstm_runs/train-{now}")
test_writer = SummaryWriter(f"lstm_runs/test-{now}")


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


num_epochs = 100
progress = train(
    lstm,
    loss=mean_squared_error,
    iterator=train_iterator,
    num_epochs=num_epochs,
    lr=0.001,
)

for epoch, loss, lstm in tqdm(progress, total=num_epochs):

    # check loss and accuracy every 100 epochs
    if epoch % 5 == 0:

        print(epoch, loss)
        eval_loss = 0.0
        eval_probs = []
        eval_labels = []
        for batch in test_iterator:
            test_loss = mean_squared_error(
                lstm, inputs=batch.inputs, targets=batch.targets
            )
            eval_loss += test_loss

            test_probs = lstm.predict_proba(batch.inputs)
            eval_probs.append(test_probs)

            eval_labels.append(batch.targets)

        eval_loss = eval_loss / len(test_iterator)
        eval_labels = np.vstack(eval_labels)
        eval_probs = np.vstack(eval_probs)

        test_acc = accuracy(eval_labels, eval_probs)
        print(f"Test Accuracy: {test_acc}")
        print(f"Test Loss: {eval_loss}")
        test_writer.add_scalar("accuracy", float(test_acc), epoch)
        test_writer.add_scalar("loss", float(eval_loss), epoch)
        train_writer.add_embedding(
            onp.array(lstm.embeddings.embedding_matrix),
            metadata=vocab,
            global_step=epoch,
        )

        test_writer.flush()
        lstm.save(f"lstm_runs/imdb_lstm_{epoch}.pkl", overwrite=True)

        if test_acc >= 0.99:
            break

    train_writer.add_scalar("loss", float(loss), epoch)
    train_writer.flush()

test_writer.close()
train_writer.close()

lstm.save("lstm_runs/imdb_lstm_final.pkl", overwrite=True)
