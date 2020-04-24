#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import jax.numpy as np
import jax.random as jxr
import numpy as onp
import plotly.graph_objects as go
from tqdm.autonotebook import tqdm

from colin_net.data import BatchIterator
from colin_net.layers import Linear, Softmax, Tanh
from colin_net.loss import mean_sqaured_error
from colin_net.nn import NeuralNet
from colin_net.train import train


# Create Input Data and True Labels
inputs = onp.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = onp.array([[0, 1], [1, 0], [1, 0], [0, 1]])


# Generate seed for Reproducible Random Numbers
key = jxr.PRNGKey(42)


# Create NeuralNet Instance
net = NeuralNet(
    [
        Linear.initialize(input_size=2, output_size=3, key=key),
        Tanh(),
        Linear.initialize(input_size=3, output_size=2, key=key),
        Softmax(),
    ]
)


# Create an iterator over the input data
iterator = BatchIterator(inputs, targets)


# define accuracy calculation
def accuracy(actual, predicted):
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


# Start training process

num_epochs = 500
progress = train(
    net, num_epochs=num_epochs, iterator=iterator, loss=mean_sqaured_error, lr=1.0
)

points = []
for i, (epoch, loss, net) in enumerate(tqdm(progress, total=num_epochs)):

    # check loss and accuracy every 5 epochs
    if i % 5 == 0:
        print(epoch, loss)
        predicted = net(inputs)

        if accuracy(targets, predicted) >= 0.99:
            print("Achieved Perfect Prediction!")
            points.append([epoch, loss])
            break
    points.append([epoch, loss])


# Display Predictions
predicted = net(inputs)
print(targets, predicted, np.argmax(predicted, axis=1))

print("Accuracy: ", accuracy(targets, predicted))


# Plott Loss Curve
points_array = np.array(points)

trace = [
    go.Scattergl(
        x=points_array[:, 0], y=points_array[:, 1], name="train loss", opacity=0.5
    )
]


layout = go.Layout(
    title="Train Loss Over Time",
    xaxis=dict(title="Number of updates"),
    yaxis=dict(title="Loss"),
)

fig = go.Figure(data=trace, layout=layout)

fig.show()
