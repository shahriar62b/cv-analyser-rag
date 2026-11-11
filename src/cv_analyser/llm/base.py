from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def generate_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Ask the model a question and return its answer validated as `schema`.

        Raises LLMError if the model is unreachable or its output does not fit
        the schema.
        """
        ...
