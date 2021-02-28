from pydantic import BaseModel
from jax.tree_util import register_pytree_node
from typing import Any, Sequence, Dict, Tuple
from lorax.tensor import Tensor


class Parameter(BaseModel):

    __root__: Tensor

    class Config:
        allow_mutation = False

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            **super().dict(),
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"Parameter shape={self.__root__.shape}"

    def __str__(self) -> str:
        return self.json()

    @property
    def value(self) -> Tensor:
        return self.__root__

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "Parameter":
        return cls(__root__=tensor)


def _tree_flatten(param: Parameter) -> Tuple[Tuple[Tensor], None]:
    return (param.__root__,), None


def _tree_unflatten(aux: Any, params: Sequence[Tensor]) -> Parameter:
    return Parameter.construct(__root__=params[0])


register_pytree_node(Parameter, _tree_flatten, _tree_unflatten)
