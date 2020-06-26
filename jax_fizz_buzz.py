"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number
"""
import datetime
from typing import List

import jax.numpy as np
import numpy as onp
from jax import random
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from colin_net.data import BatchIterator
from colin_net.loss import mean_squared_error
from colin_net.nn import MLP
from colin_net.tensor import Tensor
from colin_net.train import train

key = random.PRNGKey(42)


def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]


inputs = np.array([binary_encode(x) for x in range(101, 1024)])

targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])

test_X = np.array([binary_encode(x) for x in range(1, 101)])
test_y = np.array([fizz_buzz_encode(x) for x in range(1, 101)])

now = datetime.datetime.now().isoformat()

train_writer = SummaryWriter(f"runs/train-{now}")
test_writer = SummaryWriter(f"runs/test-{now}")

net = MLP.create_mlp(
    input_dim=10, output_dim=4, hidden_dim=50, key=key, num_hidden=2, dropout_keep=None,
)

train_writer.add_text("Net", f"```{net.json()}```")

iterator = BatchIterator(inputs=inputs, targets=targets, key=key)


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


num_epochs = 5000

progress = train(
    net, loss=mean_squared_error, iterator=iterator, num_epochs=num_epochs, lr=0.1,
)

for i, (epoch, loss, net) in enumerate(tqdm(progress, total=num_epochs)):

    # check loss and accuracy every 100 epochs
    if i % 100 == 0:
        net = net.to_eval()
        print(epoch, loss)
        predicted = net.predict_proba(inputs)
        acc_metric = accuracy(targets, predicted)
        test_predicted = net.predict_proba(test_X)
        test_acc = accuracy(test_y, test_predicted)
        print(f"Train Accuracy: {acc_metric}")
        print(f"Test Accuracy: {test_acc}")
        test_writer.add_scalar("accuracy", float(test_acc), i)
        train_writer.add_scalar("accuracy", float(acc_metric), i)
        train_writer.add_histogram("Layer1-w", onp.array(net.layers[0].w), i)
        train_writer.add_histogram("Layer1-b", onp.array(net.layers[0].b), i)
        train_writer.add_histogram("Layer2-w", onp.array(net.layers[1].w), i)
        train_writer.add_histogram("Layer2-b", onp.array(net.layers[1].b), i)
        if test_acc >= 0.99:
            break
        net = net.to_train()

    net = net.to_eval()
    eval_loss = mean_squared_error(net, test_X, test_y)
    net = net.to_train()

    test_writer.add_scalar("loss", float(eval_loss), i)
    train_writer.add_scalar("loss", float(loss), i)


net.save("jax_fizz_buzz.pkl", overwrite=True)
net = net.to_eval()
test_predictions = net.predict_proba(test_X)

test_accuracy = accuracy(test_y, test_predictions)
test_writer.add_pr_curve("PR", onp.array(test_y), onp.array(test_predictions))


print(f"Test Accuracy: {test_accuracy}")
test_writer.close()
train_writer.close()
