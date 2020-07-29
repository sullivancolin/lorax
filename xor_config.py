"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import json

import jax.numpy as np
import wandb
from tqdm.autonotebook import tqdm

from colin_net.metrics import accuracy
from colin_net.train import Experiment, wandb_log, wandb_notes

# Create Input Data and True Labels
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

with open("mlp_experiment_defaults.json") as infile:
    config = json.load(infile)


wandb.init(project="colin_net_xor", config=config)
config = wandb.config


experiment = Experiment(**config)

print(json.dumps(experiment.dict(), indent=4))

update_generator = experiment.train(
    inputs, targets, inputs, targets, iterator_type="batch_iterator"
)

bar = tqdm(total=experiment.global_step)
for update_state in update_generator:
    if update_state.step == 1:
        markdown = f"```json\n{update_state.model.json()}\n```"
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
