from typing import Any
from app.executors.base import ExecutionContext, Executor, HumanInputRequired


class LLMExecutor(Executor):
    def __init__(self, client: Any | None = None):
        self.client = client

    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        if self.client is None:
            return {"status": "completed", "content": task.get("input", {}), "executor": "llm-fallback", "trace": {"reason": "No external LLM client configured; payload preserved."}}
        response = await self.client.ainvoke(task)
        return {"status": "completed", "content": response.content, "executor": "llm"}


class MLExecutor(Executor):
    def __init__(self, model_loader: Any | None = None):
        self.model_loader = model_loader

    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        if self.model_loader is None:
            features = task.get("features", [])
            return {"status": "completed", "prediction": None, "features_received": len(features), "executor": "ml-fallback"}
        model = self.model_loader(context.subject_id)
        prediction = model.predict(task["features"])
        return {"status": "completed", "prediction": prediction.tolist(), "executor": "ml"}


class HumanExecutor(Executor):
    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        raise HumanInputRequired({"status": "waiting_for_human", "execution_id": context.execution_id, "node_id": context.node_id, "review_payload": task})


class ExecutorRegistry:
    def __init__(self):
        self._executors: dict[str, Executor] = {"llm": LLMExecutor(), "ml": MLExecutor(), "human": HumanExecutor(), "tool": LLMExecutor()}

    def register(self, kind: str, executor: Executor) -> None:
        self._executors[kind] = executor

    def get(self, kind: str) -> Executor:
        if kind not in self._executors:
            raise KeyError(f"No executor registered for {kind}")
        return self._executors[kind]
