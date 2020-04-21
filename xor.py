"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import jax.numpy as np

from colin_net.train import train
from colin_net.nn import NeuralNet
from colin_net.layers import Linear, Tanh
from colin_net.data import BatchIterator
from colin_net.loss import mean_sqaured_error
import jax.random as jxr

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

key = jxr.PRNGKey(42)

net = NeuralNet(
    [
        Linear(w=jxr.normal(key, shape=(2, 2)), b=np.zeros(shape=(2,))),
        Tanh(),
        Linear(w=jxr.normal(key, shape=(2, 2)), b=np.zeros(shape=(2,))),
    ]
)

iterator = BatchIterator()

train(net, inputs, targets, num_epochs=500, iterator=iterator, loss=mean_sqaured_error)

for x, y in zip(inputs, targets):
    predicted = net(x)
    print(x, predicted, y)
