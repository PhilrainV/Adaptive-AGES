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

def _find_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_value(child, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_value(child, key)
            if found is not None:
                return found
    return None


def _skill_text(skills: list[Any]) -> str:
    rendered = []
    for skill in skills:
        if isinstance(skill, str):
            rendered.append(f"- {skill}")
        elif isinstance(skill, dict):
            name = skill.get("name", "unnamed_skill")
            instruction = skill.get("instructions") or skill.get("description") or ""
            rendered.append(f"- {name}: {instruction}")
    return "\n".join(rendered)


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
        system_prompt = context.config.get("system_prompt", "你是严谨的任务执行智能体。")
        skills = _skill_text(context.config.get("skills", []))
        if skills:
            system_prompt = f"{system_prompt}\n\n本节点可用 Skills：\n{skills}"
        response = await client.ainvoke([
            SystemMessage(content=system_prompt),
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
        raise HumanInputRequired({
            "status": "waiting_for_human",
            "execution_id": context.execution_id,
            "node_id": context.node_id,
            "instruction": context.config.get("instruction", ""),
            "approval_criteria": context.config.get("approval_criteria", ""),
            "response_schema": context.config.get("response_schema", {}),
            "review_payload": task,
        })


class ToolExecutor(Executor):
    async def execute(self, task: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        operation = context.config.get("operation", "transform")
        payload = task.get("input", {})
        upstream = task.get("upstream", {})
        if operation == "validate_learning_evidence":
            responses = payload.get("responses") or _find_value(upstream, "responses")
            issues: list[dict[str, Any]] = []
            valid: list[dict[str, Any]] = []
            if not isinstance(responses, list):
                responses = []
                issues.append({"field": "responses", "reason": "必须是非空数组"})
            for index, row in enumerate(responses):
                if not isinstance(row, dict):
                    issues.append({"index": index, "reason": "记录必须是对象"})
                    continue
                if not row.get("knowledge_point"):
                    issues.append({"index": index, "field": "knowledge_point", "reason": "缺失"})
                    continue
                if "correct" not in row and "score" not in row:
                    issues.append({"index": index, "field": "correct|score", "reason": "缺失"})
                    continue
                valid.append(row)
            return {
                "status": "completed",
                "validated_responses": valid,
                "data_quality": {"valid": bool(valid) and not issues, "issues": issues},
                "target_mastery": float(payload.get("target_mastery", .8)),
                "executor": "builtin-tool",
            }
        if operation == "validate_exercise_set":
            exercises = _find_value(upstream, "exercises") or payload.get("exercises") or []
            required = ("question", "answer", "explanation", "knowledge_point")
            issues = []
            seen: set[str] = set()
            valid = []
            for index, item in enumerate(exercises if isinstance(exercises, list) else []):
                missing = [field for field in required if not isinstance(item, dict) or not item.get(field)]
                fingerprint = str(item.get("question", "")).strip().lower() if isinstance(item, dict) else ""
                if missing:
                    issues.append({"index": index, "missing": missing})
                elif fingerprint in seen:
                    issues.append({"index": index, "reason": "duplicate_question"})
                else:
                    seen.add(fingerprint)
                    valid.append(item)
            return {
                "status": "completed",
                "validated_exercises": valid,
                "quality": {"valid": bool(valid) and not issues, "issues": issues},
                "executor": "builtin-tool",
            }
        if operation == "score_learning_responses":
            exercises = _find_value(upstream, "validated_exercises") or payload.get("exercises") or []
            responses = _find_value(upstream, "learner_responses") or payload.get("learner_responses") or []
            answer_key = {
                str(item.get("id", index)): item
                for index, item in enumerate(exercises)
                if isinstance(item, dict)
            }
            totals: dict[str, list[float]] = {}
            item_scores = []
            for index, response in enumerate(responses if isinstance(responses, list) else []):
                if not isinstance(response, dict):
                    continue
                item_id = str(response.get("exercise_id", response.get("id", index)))
                exercise = answer_key.get(item_id, {})
                expected = str(exercise.get("answer", "")).strip().casefold()
                actual = str(response.get("answer", "")).strip().casefold()
                score = float(bool(expected) and actual == expected)
                knowledge = str(exercise.get("knowledge_point", "unknown"))
                totals.setdefault(knowledge, []).append(score)
                item_scores.append({"exercise_id": item_id, "knowledge_point": knowledge, "score": score})
            by_knowledge = {
                name: round(sum(values) / len(values), 4)
                for name, values in totals.items()
            }
            overall = round(
                sum(item["score"] for item in item_scores) / max(1, len(item_scores)),
                4,
            )
            return {
                "status": "completed",
                "item_scores": item_scores,
                "score_by_knowledge_point": by_knowledge,
                "overall_score": overall,
                "executor": "builtin-tool",
            }
        if operation == "evaluate_threshold":
            score = float(_find_value(upstream, "overall_score") or payload.get("score", 0))
            threshold = float(payload.get("threshold", context.config.get("parameters", {}).get("threshold", .8)))
            return {
                "status": "completed",
                "score": score,
                "threshold": threshold,
                "passed": score >= threshold,
                "executor": "builtin-tool",
            }
        if operation == "validate_payload":
            required = context.config.get("parameters", {}).get("required_fields", [])
            missing = [
                field for field in required
                if str(field).split(".", 1)[0].replace("[]", "") not in payload
            ]
            return {
                "status": "completed",
                "valid": not missing,
                "missing_fields": missing,
                "content": payload,
                "executor": "builtin-tool",
            }
        return {
            "status": "completed",
            "content": task,
            "operation": operation,
            "executor": context.config.get("connector", "builtin-tool"),
        }


class ExecutorRegistry:
    def __init__(self, model_config: dict[str, Any] | None = None):
        self._executors: dict[str, Executor] = {
            "llm": LLMExecutor(model_config=model_config),
            "ml": MLExecutor(),
            "human": HumanExecutor(),
            "tool": ToolExecutor(),
        }

    def register(self, kind: str, executor: Executor) -> None:
        self._executors[kind] = executor

    def get(self, kind: str) -> Executor:
        if kind not in self._executors:
            raise KeyError(f"No executor registered for {kind}")
        return self._executors[kind]
