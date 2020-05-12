"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number
"""
from typing import List

import jax.numpy as np
import plotly.graph_objects as go
from jax import random
from tqdm.autonotebook import tqdm

from colin_net.data import BatchIterator
from colin_net.loss import mean_sqaured_error
from colin_net.nn import FeedForwardNet
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

net = FeedForwardNet.create_mlp(
    input_dim=10, output_dim=4, hidden_dim=50, key=key, num_hidden=2, dropout_keep=None,
)

iterator = BatchIterator(inputs=inputs, targets=targets)


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


num_epochs = 5000

progress = train(
    net,
    key=key,
    loss=mean_sqaured_error,
    iterator=iterator,
    num_epochs=num_epochs,
    lr=0.1,
)

points = []
eval_points = []
for i, (epoch, loss, net) in enumerate(tqdm(progress, total=num_epochs)):

    # check loss and accuracy every 100 epochs
    if i % 100 == 0:
        net.eval()
        print(epoch, loss)
        keys = random.split(key, num=inputs.shape[0])
        predicted = net.predict_proba(inputs, keys)
        acc_metric = accuracy(targets, predicted)
        test_predicted = net.predict_proba(test_X, keys[: len(test_X)])
        test_acc = accuracy(test_y, test_predicted)
        print(f"Train Accuracy: {acc_metric}")
        print(f"Test Accuracy: {test_acc}")
        if acc_metric >= 0.99:
            break
        net.train()

    points.append([epoch, loss])
    net.eval()
    eval_loss = mean_sqaured_error(net, keys[: len(test_X)], test_X, test_y)
    net.train()
    eval_points.append(eval_loss)

net.eval()

net.save("jax_fizz_buzz.pkl", overwrite=True)
keys = random.split(key, num=test_X.shape[0])

test_predictions = net.predict_proba(test_X, keys)

test_accuracy = accuracy(test_y, test_predictions)

print(f"Test Accuracy: {test_accuracy}")

# Plott Loss Curve
points_array = np.array(points)

trace = [
    go.Scattergl(
        x=points_array[:, 0], y=points_array[:, 1], name="train loss", opacity=0.5,
    ),
    go.Scattergl(x=points_array[:, 0], y=eval_points, name="test loss", opacity=0.5,),
]

layout = go.Layout(
    title="FizzBuzz Loss Over Time",
    xaxis=dict(title="epochs"),
    yaxis=dict(title="Loss"),
    width=800,
    height=700,
)

fig = go.Figure(data=trace, layout=layout)

fig.write_html("fizzbuzz_loss_curve.html")
