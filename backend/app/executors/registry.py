import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.executors.base import ExecutionContext, Executor, HumanInputRequired


class LLMExecutor(Executor):
    def __init__(self, client: Any | None = None, model_config: dict[str, Any] | None = None):
        self.client = client
        self.model_config = model_config or {}

    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        client = self.client
        if client is None and self.model_config.get("api_key"):
            client = ChatOpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config.get("base_url") or None,
                model=self.model_config.get("model") or "gpt-4.1-mini",
                temperature=float(context.config.get("temperature", self.model_config.get("temperature", .2))),
                timeout=60,
            )
        if client is None:
            return {
                "status": "completed",
                "content": task.get("input", {}),
                "executor": "llm-fallback",
                "trace": {"reason": "未配置模型 API；已保留输入。请在系统设置中配置后重新运行。"},
            }
        template = context.config.get("prompt_template", "用户输入：{input}\n上游结果：{upstream}")
        prompt = template.replace("{input}", json.dumps(task.get("input", {}), ensure_ascii=False)).replace(
            "{upstream}", json.dumps(task.get("upstream", {}), ensure_ascii=False)
        )
        response = await client.ainvoke([
            SystemMessage(content=context.config.get("system_prompt", "你是严谨的任务执行智能体。")),
            HumanMessage(content=prompt),
        ])
        return {"status": "completed", "content": response.content, "executor": "llm"}


class MLExecutor(Executor):
    def __init__(self, model_loader: Any | None = None):
        self.model_loader = model_loader

    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        if self.model_loader is None:
            features = task.get("input", {}).get("features", [])
            return {
                "status": "completed", "prediction": None, "features_received": len(features),
                "executor": "ml-export-runtime",
                "trace": {"reason": "自定义 ML 代码仅在隔离的导出包中执行，平台服务端不会直接执行任意代码。"},
            }
        model = self.model_loader(context.subject_id)
        prediction = model.predict(task["features"])
        return {"status": "completed", "prediction": prediction.tolist(), "executor": "ml"}


class HumanExecutor(Executor):
    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        raise HumanInputRequired({"status": "waiting_for_human", "execution_id": context.execution_id, "node_id": context.node_id, "review_payload": task})


class PassthroughExecutor(Executor):
    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        return {"status": "completed", "content": task, "executor": context.config.get("connector", "passthrough")}


class ExecutorRegistry:
    def __init__(self, model_config: dict[str, Any] | None = None):
        self._executors: dict[str, Executor] = {
            "llm": LLMExecutor(model_config=model_config),
            "ml": MLExecutor(),
            "human": HumanExecutor(),
            "tool": PassthroughExecutor(),
        }

    def register(self, kind: str, executor: Executor) -> None:
        self._executors[kind] = executor

    def get(self, kind: str) -> Executor:
        if kind not in self._executors:
            raise KeyError(f"No executor registered for {kind}")
        return self._executors[kind]
