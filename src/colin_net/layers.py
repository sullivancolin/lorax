"""
Our neural nets will be made up of layers.
Each layer needs to pass the inputs forward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from enum import Enum
from typing import Any, Iterable, List, Tuple

import jax.numpy as np
from jax import jit, nn, ops, random

from colin_net.base import PyTreeLike
from colin_net.tensor import Tensor

LinearTuple = Tuple[Tensor, Tensor]


class Mode(str, Enum):
    """Allowed values for Dropout Mode"""

    train = "train"
    eval = "eval"


INITIALIZERS = {
    "normal": nn.initializers.normal(stddev=1.0),
    "glorot_normal": nn.initializers.glorot_normal(),
    "lecun_normal": nn.initializers.lecun_normal(),
}


class InitializerEnum(str, Enum):
    normal = "normal"
    glorot_normal = "glorot_normal"
    lecun_normal = "lecun_normal"


class Layer(PyTreeLike, is_abstract=True):
    """Abstract Class for Layers. Enforces subclasses to implement
    __call__, tree_flatten, tree_unflatten and registered as Pytree"""

    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__repr__()


class ActivationEnum(str, Enum):
    tanh = "tanh"
    relu = "relu"
    leaky_relu = "leaky_relu"
    selu = "selu"
    sigmoid = "sigmoid"
    softmax = "softmax"


class ActivationLayer(Layer, is_abstract=True):
    """Abstract Class for Activation Layers."""

    def tree_flatten(self) -> Tuple[List[None], None]:
        return ([None], None)

    @classmethod
    def tree_unflatten(cls, aux: Any, data: Iterable[Any]) -> "ActivationLayer":
        return cls()

    @staticmethod
    def initialize(activation: ActivationEnum) -> "ActivationLayer":
        return ACTIVATIONS[activation]()

    def __repr__(self) -> str:
        return f"<ActivationLayer {self.__class__.__name__}>"


class Linear(Layer):
    """Dense Linear Layer.
    Computes output = np.dot(w, inputs) + b"""

    def __init__(self, w: Tensor, b: Tensor) -> None:
        self.w = w
        self.b = b

    def __repr__(self) -> str:
        return f"<LinearLayer w={self.w.shape}, b={self.b.shape}>"

    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """outputs = np.dot(w, inputs) + b in single instance notation."""

        return np.dot(self.w, inputs) + self.b

    @classmethod
    def initialize(
        cls,
        *,
        input_size: int,
        output_size: int,
        key: Tensor,
        initializer: InitializerEnum = InitializerEnum.normal,
    ) -> "Linear":
        """Factory for new Linear from input and output dimentsions"""
        if initializer not in InitializerEnum.__members__:
            raise ValueError(
                f"initializer: {initializer} not in {InitializerEnum.__members__.values()}"
            )
        return cls(
            w=INITIALIZERS[initializer](key, shape=(output_size, input_size)),
            b=np.zeros(shape=(output_size,)),
        )

    def tree_flatten(self) -> Tuple[LinearTuple, None]:
        return ((self.w, self.b), None)

    @classmethod
    def tree_unflatten(cls, aux: Any, params: LinearTuple) -> "Linear":
        return cls(*params)


class Dropout(Layer):
    """Dropout Layer. If in train mode, keeps input activations at given probability rate,
    otherwise returns inputs directly"""

    def __init__(self, keep: float = 0.5, mode: str = Mode.train) -> None:
        self.keep = keep
        if mode not in Mode.__members__:
            raise ValueError(f"mode: {mode} not in {Mode.__members__.values()}")
        self.mode = mode

    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """If in train mode, keeps input activations at rate,
        otherwise returns directly"""

        if self.mode == Mode.eval:
            return inputs

        key = kwargs.get("key", None)
        if key is None and self.mode != Mode.eval:
            msg = (
                "Dropout layer requires __call__ to be called with a PRNG key "
                "argument. That is, instead of `__call__(inputs)`, use "
                "it like `__call__(inputs, key)` where `key` is a "
                "jax.random.PRNGKey value."
            )
            raise ValueError(msg)
        mask = random.bernoulli(key, self.keep, inputs.shape)
        return np.where(mask, inputs / self.keep, 0)

    def __repr__(self) -> str:
        return f"<Dropout keep={self.keep}, mode={self.mode}>"

    def tree_flatten(self) -> Tuple[List[None], Tuple[float, str]]:
        return ([None], (self.keep, self.mode))

    @classmethod
    def tree_unflatten(cls, aux: Tuple[float, str], params: List[Any]) -> "Dropout":
        return cls(*aux)


class Embedding(Layer):
    def __init__(self, embedding_matrix: Tensor) -> None:
        self.embedding_matrix = embedding_matrix

    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return self.embedding_matrix[inputs]

    @classmethod
    def initialize(
        cls,
        vocab_size: int,
        hidden_dim: int,
        key: Tensor,
        initializer: InitializerEnum = InitializerEnum.glorot_normal,
    ) -> "Embedding":
        vectors = INITIALIZERS[initializer](key, shape=(vocab_size, hidden_dim))
        vectors = ops.index_update(vectors, ops.index[0, :], 0.0)
        return cls(vectors)

    def tree_flatten(self) -> Tuple[Tuple[Tensor], None]:
        return (self.embedding_matrix,), None

    @classmethod
    def tree_unflatten(cls, aux_data: Any, values: Tuple[Tensor]) -> "Embedding":
        return cls(*values)


class LSTMCell(Layer):
    def __init__(
        self,
        Wf: Tensor,
        bf: Tensor,
        Wi: Tensor,
        bi: Tensor,
        Wc: Tensor,
        bc: Tensor,
        Wo: Tensor,
        bo: Tensor,
    ) -> None:
        self.Wf = Wf
        self.bf = bf
        self.Wi = Wi
        self.bi = bi
        self.Wc = Wc
        self.bc = bc
        self.Wo = Wo
        self.bo = bo

    @classmethod
    def initialize(cls, input_dim: int, hidden_dim: int, key: Tensor) -> "LSTMCell":

        Wf_key, Wi_key, Wc_key, Wo_key = random.split(key, num=4)

        concat_size = input_dim + hidden_dim

        Wf = INITIALIZERS["glorot_normal"](Wf_key, shape=(hidden_dim, concat_size))
        bf = np.zeros(shape=(hidden_dim,))

        Wi = INITIALIZERS["glorot_normal"](Wi_key, shape=(hidden_dim, concat_size))
        bi = np.zeros(shape=(hidden_dim,))

        Wc = INITIALIZERS["glorot_normal"](Wc_key, shape=(hidden_dim, concat_size))
        bc = np.zeros(shape=(hidden_dim,))

        Wo = INITIALIZERS["glorot_normal"](Wo_key, shape=(hidden_dim, concat_size))
        bo = np.zeros(shape=(hidden_dim,))

        return cls(Wf, bf, Wi, bi, Wc, bc, Wo, bo)

    @jit
    def __call__(
        self, state: Tuple[Tensor, Tensor], inputs: Tensor, **kwargs: Any
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:

        h_prev, c_prev = state
        concat_vec = np.hstack((inputs, h_prev))

        f = nn.sigmoid(np.dot(self.Wf, concat_vec) + self.bf)
        i = nn.sigmoid(np.dot(self.Wi, concat_vec) + self.bi)
        C_bar = np.tanh(np.dot(self.Wc, concat_vec) + self.bc)

        c = f * c_prev + i * C_bar
        o = nn.sigmoid(np.dot(self.Wo, concat_vec) + self.bo)
        h = o * np.tanh(c)

        # hiddent state vector is copied as out_put
        return (h, c), h

    def tree_flatten(self) -> Tuple[List[Tensor], None]:
        return (
            [self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo],
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data: Any, params: List[Tensor]) -> "LSTMCell":
        return cls(*params)


class Tanh(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return np.tanh(inputs)


class Relu(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.relu(inputs)


class LeakyRelu(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.leaky_relu(inputs)


class Selu(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.selu(inputs)


class Sigmoid(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.sigmoid(inputs)


class Softmax(ActivationLayer):
    @jit
    def __call__(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        return nn.softmax(inputs)


ACTIVATIONS = {
    "tanh": Tanh,
    "relu": Relu,
    "leaky_relu": LeakyRelu,
    "selu": Selu,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
}
