"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import jax.numpy as np

from colin_net.config import Experiment
from colin_net.tensor import Tensor

# Create Input Data and True Labels
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


config = {
    "experiment_name": "xor_runs",
    "net_config": {
        "output_dim": 2,
        "input_dim": 2,
        "hidden_dim": 5,
        "num_hidden": 5,
        "activation": "tanh",
        "dropout_keep": None,
    },
    "random_seed": 42,
    "loss": "mean_squared_error",
    "regularization": None,
    "optimizer": "adam",
    "iterator_type": "batch_iterator",
    "learning_rate": 0.001,
    "batch_size": 4,
    "global_step": 5000,
    "log_every": 10,
}


experiment = Experiment(**config)


# define accuracy calculation
def accuracy(actual: Tensor, predicted: Tensor) -> float:
    return np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))


for update_state in experiment.train(inputs, targets, inputs, targets):
    if update_state.iteration % experiment.log_every == 0:
        net = update_state.net
        train_predicted = net.predict_proba(inputs)
        train_accuracy = float(accuracy(targets, train_predicted))
        net = net.to_eval()
        predicted = net.predict_proba(inputs)
        acc_metric = float(accuracy(targets, predicted))
        update_state.test_writer.add_scalar(
            "accuracy", acc_metric, update_state.iteration
        )
        update_state.train_writer.add_scalar(
            "accuracy", train_accuracy, update_state.iteration
        )
        print(f"Accuracy: {acc_metric}")
        update_state.train_writer.flush()
        update_state.test_writer.flush()
        if acc_metric >= 0.99:
            print("Achieved Perfect Prediction!")
            break
        net = net.to_train()


final_net = update_state.net
final_net.save(f"{experiment.experiment_name}/final_model.pkl", overwrite=True)

# Display Predictions
final_net = final_net.to_eval()
probabilties = final_net.predict_proba(inputs)
for gold, prob, pred in zip(targets, probabilties, np.argmax(probabilties, axis=1)):

    print(gold, prob, pred)

accuracy_score = float(accuracy(targets, probabilties))
print("Accuracy: ", accuracy_score)
