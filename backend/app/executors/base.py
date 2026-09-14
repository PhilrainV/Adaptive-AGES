from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionContext:
    execution_id: str
    node_id: str
    subject_id: str
    config: dict[str, Any]


class HumanInputRequired(RuntimeError):
    def __init__(self, payload: dict[str, Any]):
        super().__init__("Human input required")
        self.payload = payload


class Executor(ABC):
    @abstractmethod
    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        raise NotImplementedError
