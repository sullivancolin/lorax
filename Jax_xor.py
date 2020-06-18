"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import datetime

import jax.numpy as np
from jax import random
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from colin_net import MLP
from colin_net.data import BatchIterator
from colin_net.loss import mean_squared_error
from colin_net.tensor import Tensor
from colin_net.train import train

now = datetime.datetime.now().isoformat()
writer = SummaryWriter(f"xor_runs/train-{now}")

# Create Input Data and True Labels
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


# Generate seed for Reproducible Random Numbers
key = random.PRNGKey(42)

key, subkey = random.split(key)

# Create NeuralNet Instance
net = MLP.create_mlp(
    input_dim=2, output_dim=2, hidden_dim=2, key=subkey, dropout_keep=0.8, num_hidden=2,
)

key, subkey = random.split(key)
# Create an iterator over the input data
iterator = BatchIterator(inputs, targets, subkey)


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


# Start training process
num_epochs = 5000
progress = train(
    net, num_epochs=num_epochs, iterator=iterator, loss=mean_squared_error, lr=0.1,
)

for i, (epoch, loss, net) in enumerate(tqdm(progress, total=num_epochs)):

    # checks accuracy every 50 epochs
    if i % 5 == 0:
        print(epoch, loss)
        net.eval()
        predicted = net.predict_proba(inputs)
        acc_metric = float(accuracy(targets, predicted))
        writer.add_scalar("train_accuracy", acc_metric, i)
        print(f"Accuracy: {acc_metric}")
        if acc_metric >= 0.99:
            print("Achieved Perfect Prediction!")
            break
        net.train()
    writer.add_scalar("train_loss", float(loss), i)

net.save("xor_model.pkl", overwrite=True)
writer.close()

# Display Predictions
probabilties = net.predict_proba(inputs)
for gold, prob, pred in zip(targets, probabilties, np.argmax(probabilties, axis=1)):

    print(gold, prob, pred)

print("Accuracy: ", accuracy(targets, probabilties))
