"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import jax.numpy as np

from colin_net.train import train
from colin_net.nn import NeuralNet
from colin_net.layers import Linear, Tanh, Sigmoid, Softmax
from colin_net.data import BatchIterator
from colin_net.loss import mean_sqaured_error
import jax.random as jxr

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

key = jxr.PRNGKey(42)

net = NeuralNet(
    [
        Linear(w=jxr.normal(key, shape=(2, 2)), b=jxr.normal(key, shape=(2,))),
        Tanh(),
        Linear(w=jxr.normal(key, shape=(2, 2)), b=jxr.normal(key, shape=(2,))),
        Softmax(),
    ]
)

iterator = BatchIterator(shuffle=False)

train(net, inputs, targets, num_epochs=5000, iterator=iterator, loss=mean_sqaured_error)


def accuracy(actual, predicted):
    np.sum(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1)) / len(actual)


for x, y in zip(inputs, targets):
    predicted = net(x)
    print(x, predicted, y)


predicted = np.array([net(x) for x in inputs])

print(accuracy(targets, predicted))
