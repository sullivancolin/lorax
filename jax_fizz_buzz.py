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
import jax.random as jxr
import plotly.graph_objects as go
from jax import nn
from tqdm.autonotebook import tqdm

from colin_net.data import BatchIterator
from colin_net.layers import Linear, Relu, Tanh
from colin_net.loss import cross_entropy_loss, mean_sqaured_error
from colin_net.nn import NeuralNet
from colin_net.train import train

key = jxr.PRNGKey(42)


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

net = NeuralNet(
    [
        Linear.initialize(input_size=10, output_size=50, key=key),
        Tanh(),
        Linear.initialize(input_size=50, output_size=4, key=key),
    ]
)

iterator = BatchIterator(inputs=inputs, targets=targets)


# define accuracy calculation
def accuracy(actual, predicted):
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


num_epochs = 5000

progress = train(
    net, loss=mean_sqaured_error, iterator=iterator, num_epochs=num_epochs, lr=0.1
)


points = []
for i, (epoch, loss, net) in enumerate(tqdm(progress, total=num_epochs)):

    # check loss and accuracy every 100 epochs
    if i % 100 == 0:
        print(epoch, loss)
        predicted = nn.softmax(net(inputs))
        acc_metric = accuracy(targets, predicted)
        print(f"Train Accuracy: {acc_metric}")

    points.append([epoch, loss])


for x in range(1, 101):
    predicted = nn.softmax(net.predict(np.array(binary_encode(x))))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(np.array(fizz_buzz_encode(x)))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])


test_predictions = nn.softmax(net(np.array([binary_encode(x) for x in range(1, 101)])))
test_labels = np.array([fizz_buzz_encode(x) for x in range(1, 101)])

test_accuracy = accuracy(test_labels, test_predictions)

print(f"Test Accuracy: {test_accuracy}")

# Plott Loss Curve
points_array = np.array(points)

trace = [
    go.Scattergl(
        x=points_array[:, 0], y=points_array[:, 1], name="train loss", opacity=0.5
    )
]


layout = go.Layout(
    title="FizzBuzz Train Loss Over Time",
    xaxis=dict(title="Number of updates"),
    yaxis=dict(title="Loss"),
    width=600,
    height=500,
)

fig = go.Figure(data=trace, layout=layout)

fig.write_html("fizzbuzz_loss_curve.html")
