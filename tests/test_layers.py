import jax.numpy as np
import pytest
from numpy import testing

from lorax.models import MLP
from lorax.nn.layers import Dropout, Embedding
from lorax.rng import RNG
from lorax.tensor import Tensor


@pytest.fixture
def single_input() -> Tensor:
    return np.ones(shape=(2,))


@pytest.fixture
def batched_inputs(single_input: Tensor) -> Tensor:
    return np.tile(single_input, (4, 1))


@pytest.fixture
def rng() -> RNG:
    return RNG.from_seed(42)


@pytest.fixture
def dropout(rng: RNG) -> Dropout:
    d = Dropout(keep=0.5, mode="train")
    return d.initialize(rng)


def test_dropout(single_input: Tensor, dropout: Dropout) -> None:
    # with disable_jit():
    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([2.0, 0.0]))

    dropout = dropout.to_eval()

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([1.0, 1.0]))

    dropout = dropout.to_train()

    outputs = dropout(single_input)

    testing.assert_array_equal(outputs, np.array([0.0, 2.0]))


def test_mlp_dropout(batched_inputs: Tensor, rng: RNG, dropout: Dropout) -> None:

    simplenn = MLP.build(dropout)
    simplenn = simplenn.initialize(rng)

    outputs = simplenn(batched_inputs)

    testing.assert_array_equal(
        outputs, np.array([[2.0, 2.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
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


def test_embedding(embedding: Embedding) -> None:
    indices = np.array([0, 0, 0, 1, 1, 1])

    outputs = embedding(indices)

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
