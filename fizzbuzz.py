"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz"
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number
"""
import json
from typing import List

import jax.numpy as np
import wandb
from tqdm.autonotebook import tqdm

from colin_net.metrics import accuracy
from colin_net.train import Experiment, wandb_log, wandb_notes


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


with open("fizzbuzz_config.json") as infile:
    config = json.load(infile)


wandb.init(project="colin_net_fizzbuzz", config=config, save_code=True)
config = wandb.config


experiment = Experiment(**config)

print(json.dumps(experiment.dict(), indent=4))

update_generator = experiment.train(
    train_X=inputs,
    train_Y=targets,
    test_X=test_X,
    test_Y=test_y,
    iterator_type="batch_iterator",
)

bar = tqdm(total=experiment.global_step)
for update_state in update_generator:
    # if update_state.step == 1:
    #     markdown = f"# Model Definition\n```json\n{update_state.model.json()}\n```"
    #     wandb_notes(markdown)
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
probabilties = final_model.predict_proba(test_X)


for x, (gold, prob) in enumerate(zip(test_y, probabilties)):
    actual_idx = np.argmax(gold)
    predicted_idx = np.argmax(prob)

    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])


accuracy_score = float(accuracy(test_y, probabilties))
print("Accuracy: ", accuracy_score)
