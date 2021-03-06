"""
MNIST DNN classifier
"""
import json

import jax.numpy as np
import wandb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils import check_random_state
from tqdm.autonotebook import tqdm

from lorax.metrics import accuracy
from lorax.train import Experiment, wandb_notes

config = {
    "experiment_name": "mnist",
    "model_config": {
        "kind": "MLP",
        "output_dim": 10,
        "input_dim": 784,
        "hidden_sizes": [1024, 1024],
        "activation": "relu",
        "dropout_keep": None,
    },
    "random_seed": 42,
    "loss": "cross_entropy",
    "regularization": None,
    "optimizer": "adam",
    "learning_rate": 0.01,
    "batch_size": 16,
    "global_step": 2500,
    "log_every": 100,
}


wandb.init(project="colin_net_mnist", config=config, save_code=True)
config = wandb.config

experiment = Experiment.from_flattened(config)


# Create Input Data and True Labels
# Load data from https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

random_state = check_random_state(experiment.random_seed)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]

X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


print(json.dumps(experiment.dict(), indent=4))

update_generator = experiment.train(
    X_train, y_train, X_test, y_test, iterator_type="batch_iterator"
)

bar = tqdm(total=experiment.global_step)
for update_state in update_generator:
    if update_state.step == 1:
        markdown = f"{update_state.model.json()}"
        wandb_notes(markdown)
    if update_state.step % experiment.log_every == 0:
        bar.set_description(f"loss:{update_state.loss:.5f}")
    bar.update()


final_model = update_state.model

# Display Predictions
final_model = final_model.to_eval()

test_iterator = experiment.create_iterator("batch_iterator", X_test, y_test)
prob_list = []

for batch in test_iterator:
    probs = final_model.predict_proba(batch.inputs)
    prob_list.append(probs)

probabilties = np.hstack(prob_list)


accuracy_score = float(accuracy(y_test, probabilties))
print("Accuracy: ", accuracy_score)
