import jax.numpy as np
from jax import random, disable_jit
import pytest
from numpy import testing

from lorax.models import MLP
from lorax.nn import Dropout, Embedding, LSTM, Linear, Mode
from lorax.nn.functional.activations import ActivationEnum
from lorax.rng import RNG
from lorax.tensor import Tensor
from lorax.loss import mean_squared_error
from lorax.optim import SGD


@pytest.fixture
def single_input() -> Tensor:
    return np.ones(shape=(2,))


@pytest.fixture
def batched_inputs(single_input: Tensor) -> Tensor:
    return np.tile(single_input, (4, 1))


@pytest.fixture
def sequence_ids() -> Tensor:
    return np.array([0, 0, 0, 1, 1, 1])


@pytest.fixture
def batched_ids(sequence_ids: Tensor) -> Tensor:
    return np.tile(sequence_ids, (4, 1))


@pytest.fixture
def rng() -> RNG:
    return RNG.from_seed(42)


@pytest.fixture
def linear(rng: RNG) -> Linear:
    return Linear.build(
        input_dim=1, output_dim=1, activation=ActivationEnum.tanh
    ).initialize(rng)


@pytest.fixture
def dropout(rng: RNG) -> Dropout:
    return Dropout.build(keep=0.5, mode=Mode.train).initialize(rng)


def test_dropout(single_input: Tensor, dropout: Dropout) -> None:
    # with disable_jit():
    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([0.0, 2.0]))

    dropout = dropout.to_eval()

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([1.0, 1.0]))

    dropout = dropout.to_train()

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([0.0, 0.0]))


def test_mlp_dropout(batched_inputs: Tensor, rng: RNG, dropout: Dropout) -> None:

    simplenn = MLP.build(dropout)
    simplenn = simplenn.initialize(rng)

    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[0.0, 2.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    )

    simplenn = simplenn.to_eval()
    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    )

    simplenn = simplenn.to_train()
    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    )


@pytest.fixture
def embedding(rng: RNG) -> Embedding:
    return Embedding(
        embedding_matrix=np.array(
            [[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
        )
    ).initialize(rng)


def test_embedding(embedding: Embedding, sequence_ids) -> None:
    outputs = embedding(sequence_ids)

    testing.assert_array_equal(
        outputs,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
    )

    indices = np.array([2, 2, 2, 1, 1, 1])

    outputs = embedding(indices)

    testing.assert_array_equal(
        outputs,
        np.array(
            [
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
    )


def test_linear(linear: Linear, rng: RNG) -> None:
    with disable_jit():
        optimizer = SGD.initialize(linear, mean_squared_error, learning_rate=0.05)
        inputs = np.array([0.01])
        targets = np.array([0.10])

        loss = mean_squared_error(linear, inputs, targets)
        print(loss)
        for i in range(10):
            loss, linear = optimizer.step(inputs, targets)
            print(loss)

        print(loss)


def test_lstm(rng: RNG, batched_ids: Tensor, embedding: Embedding) -> None:
    lstm = LSTM.build(4, 5).initialize(rng)
    embed = embedding(batched_ids)
    assert embed.shape == (6, 4, 4)
    outs = lstm(embed)

    assert outs.shape == (4, 6, 5)
