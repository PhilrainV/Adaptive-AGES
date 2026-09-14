import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.secrets import decrypt_secret
from app.executors.base import ExecutionContext, Executor, HumanInputRequired


def _node_model_config(node_config: dict[str, Any], default_config: dict[str, Any]) -> dict[str, Any]:
    if node_config.get("use_default_model", True):
        return {**default_config, "temperature": node_config.get("temperature", default_config.get("temperature", .2))}
    api_key = None
    encrypted = node_config.get("api_key_encrypted")
    if encrypted:
        try:
            api_key = decrypt_secret(encrypted)
        except Exception:  # noqa: BLE001 - secret rotation requires the key to be saved again
            api_key = None
    return {
        "provider": node_config.get("provider", "openai-compatible"),
        "model": node_config.get("model"),
        "base_url": node_config.get("base_url"),
        "api_key": api_key,
        "temperature": node_config.get("temperature", .2),
        "modality": node_config.get("modality", "text"),
    }


def _human_message(
    prompt: str,
    task_input: dict[str, Any],
    modality: str,
    configured_urls: list[str] | None = None,
) -> HumanMessage:
    if modality != "vision":
        return HumanMessage(content=prompt)
    urls: list[str] = list(configured_urls or [])
    if isinstance(task_input.get("image_url"), str):
        urls.append(task_input["image_url"])
    if isinstance(task_input.get("image_urls"), list):
        urls.extend(str(url) for url in task_input["image_urls"] if url)
    if not urls:
        return HumanMessage(content=prompt + "\n\n未收到 image_url 或 image_urls，当前仅按文本执行。")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
    return HumanMessage(content=content)


class LLMExecutor(Executor):
    def __init__(self, client: Any | None = None, model_config: dict[str, Any] | None = None):
        self.client = client
        self.model_config = model_config or {}

    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        effective = _node_model_config(context.config, self.model_config)
        client = self.client
        if client is None and effective.get("api_key"):
            client = ChatOpenAI(
                api_key=effective["api_key"],
                base_url=effective.get("base_url") or None,
                model=effective.get("model") or "gpt-4.1-mini",
                temperature=float(effective.get("temperature", .2)),
                timeout=int(context.config.get("timeout_seconds", 60)),
            )
        if client is None:
            scope = "系统默认模型" if context.config.get("use_default_model", True) else "该 LLM 节点的独立模型"
            return {
                "status": "completed",
                "content": task.get("input", {}),
                "executor": "llm-fallback",
                "trace": {"reason": f"未配置{scope} API Key；已保留输入。请保存模型配置后重新运行。"},
            }
        template = context.config.get("prompt_template", "用户输入：{input}\n上游结果：{upstream}")
        task_input = task.get("input", {})
        prompt = template.replace("{input}", json.dumps(task_input, ensure_ascii=False)).replace(
            "{upstream}", json.dumps(task.get("upstream", {}), ensure_ascii=False)
        )
        response = await client.ainvoke([
            SystemMessage(content=context.config.get("system_prompt", "你是严谨的任务执行智能体。")),
            _human_message(
                prompt,
                task_input,
                str(effective.get("modality", context.config.get("modality", "text"))),
                context.config.get("image_urls", []),
            ),
        ])
        return {
            "status": "completed", "content": response.content, "executor": "llm",
            "model": effective.get("model"), "provider": effective.get("provider"),
        }


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
