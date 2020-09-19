"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import json

import jax.numpy as np
from tqdm.autonotebook import tqdm

import wandb
from lorax.metrics import accuracy
from lorax.train import Experiment, wandb_log, wandb_notes

# Create Input Data and True Labels
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

config = {
    "experiment_name": "xor_runs",
    "model_config": {
        "kind": "MLP",
        "output_dim": 2,
        "input_dim": 2,
        "hidden_sizes": [2],
        "activation": "tanh",
        "dropout_keep": None,
    },
    "random_seed": 42,
    "loss": "mean_squared_error",
    "regularization": None,
    "optimizer": "adam",
    "learning_rate": 0.01,
    "batch_size": 4,
    "global_step": 5000,
    "log_every": 50,
}


wandb.init(project="colin_net_xor", config=config, save_code=True)
config = wandb.config

experiment = Experiment.from_flattened(config)

print(json.dumps(experiment.dict(), indent=4))

update_generator = experiment.train(
    inputs, targets, inputs, targets, iterator_type="batch_iterator"
)

bar = tqdm(total=experiment.global_step)
for update_state in update_generator:
    if update_state.step == 1:
        markdown = f"{update_state.model.json()}"
        wandb_notes(markdown)
    if update_state.step % experiment.log_every == 0:
        model = update_state.model.to_eval()
        predicted = model.predict_proba(inputs)
        acc_metric = float(accuracy(targets, predicted)) * 100
        wandb_log({"train_accuracy": acc_metric}, step=update_state.step)
        bar.set_description(f"acc:{acc_metric:.1f}%, loss:{update_state.loss:.5f}")

        model = model.to_train()
    bar.update()


final_model = update_state.model

# Display Predictions
final_model = final_model.to_eval()
probabilties = final_model.predict_proba(inputs)
for gold, prob, pred in zip(targets, probabilties, np.argmax(probabilties, axis=1)):

    print(gold, prob, pred)

accuracy_score = float(accuracy(targets, probabilties))
print("Accuracy: ", accuracy_score)
