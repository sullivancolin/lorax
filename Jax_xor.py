"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import jax.numpy as np
import plotly.graph_objects as go
from jax import random
from tqdm.autonotebook import tqdm

from colin_net import FeedForwardNet
from colin_net.data import BatchIterator
from colin_net.loss import mean_sqaured_error
from colin_net.train import train

# Create Input Data and True Labels
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


# Generate seed for Reproducible Random Numbers
key = random.PRNGKey(42)


# Create NeuralNet Instance
net = FeedForwardNet.create_mlp(
    input_dim=2, output_dim=2, hidding_dim=2, key=key, dropout_keep=1.0, num_hidden=2,
)

# Create an iterator over the input data
iterator = BatchIterator(inputs, targets)


# define accuracy calculation
def accuracy(actual, predicted):
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


# Start training process

num_epochs = 5000
progress = train(
    net, key, num_epochs=num_epochs, iterator=iterator, loss=mean_sqaured_error, lr=0.1,
)

points = []
for i, (epoch, loss, net) in enumerate(tqdm(progress, total=num_epochs)):

    # checks accuracy every 50 epochs
    if i % 50 == 0:
        print(epoch, loss)
        net.eval()
        keys = random.split(key, num=inputs.shape[0])
        predicted = net.predict_proba(inputs, keys)
        acc_metric = accuracy(targets, predicted)
        print(f"Accuracy: {acc_metric}")
        if acc_metric >= 0.99:
            print("Achieved Perfect Prediction!")
            points.append([epoch, loss])
            break
        net.train()
    points.append([epoch, loss])

net.save("xor_model.pkl", overwrite=True)

# Display Predictions
keys = random.split(key, num=inputs.shape[0])
probabilties = net.predict_proba(inputs, keys)
for gold, prob, pred in zip(targets, probabilties, np.argmax(probabilties, axis=1)):

    print(gold, prob, pred)

print("Accuracy: ", accuracy(targets, probabilties))

# Plott Loss Curve
points_array = np.array(points)

trace = [
    go.Scattergl(
        x=points_array[:, 0], y=points_array[:, 1], name="train loss", opacity=0.5
    )
]

layout = go.Layout(
    title="Xor Train Loss Over Time",
    xaxis=dict(title="Number of updates"),
    yaxis=dict(title="Loss"),
    width=600,
    height=500,
)

fig = go.Figure(data=trace, layout=layout)

fig.write_html("xor_loss_curve.html")
