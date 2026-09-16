import io
import json
import re
import zipfile
from typing import Any

RUNNER = r'''import argparse
import importlib.util
import json
import os
import re
from pathlib import Path


def render(template, payload, upstream):
    return template.replace("{input}", json.dumps(payload, ensure_ascii=False)).replace("{upstream}", json.dumps(upstream, ensure_ascii=False))


def run_llm(node, payload, upstream):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt") from exc
    config = node.get("config", {})
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL") or None)
    skills = config.get("skills", [])
    skill_lines = []
    for skill in skills:
        if isinstance(skill, str):
            skill_lines.append(f"- {skill}")
        elif isinstance(skill, dict):
            skill_lines.append(f"- {skill.get('name', 'skill')}: {skill.get('instructions', '')}")
    system_prompt = config.get("system_prompt", "你是严谨的任务执行智能体。")
    if skill_lines:
        system_prompt += "\n\n本节点可用 Skills：\n" + "\n".join(skill_lines)
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=float(config.get("temperature", 0.2)),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render(config.get("prompt_template", "{input}\n{upstream}"), payload, upstream)},
        ],
    )
    return {"content": response.choices[0].message.content}


def run_ml(node, payload, upstream):
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", node["id"])
    path = Path("nodes") / f"{safe_id}.py"
    spec = importlib.util.spec_from_file_location(node["id"].replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(payload, upstream)

def find_value(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = find_value(child, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = find_value(child, key)
            if found is not None:
                return found
    return None


def run_tool(node, payload, upstream):
    config = node.get("config", {})
    operation = config.get("operation", "transform")
    if operation == "validate_learning_evidence":
        responses = payload.get("responses") or find_value(upstream, "responses") or []
        valid, issues = [], []
        for index, row in enumerate(responses if isinstance(responses, list) else []):
            if not isinstance(row, dict) or not row.get("knowledge_point"):
                issues.append({"index": index, "reason": "missing knowledge_point"})
            elif "correct" not in row and "score" not in row:
                issues.append({"index": index, "reason": "missing correct or score"})
            else:
                valid.append(row)
        return {
            "validated_responses": valid,
            "data_quality": {"valid": bool(valid) and not issues, "issues": issues},
            "target_mastery": float(payload.get("target_mastery", 0.8)),
        }
    if operation == "route_emotion_state":
        raw = find_value(upstream, "content")
        parsed = {}
        if isinstance(raw, str):
            candidate = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            start, end = candidate.find("{"), candidate.rfind("}")
            if start >= 0 and end > start:
                candidate = candidate[start:end + 1]
            try:
                value = json.loads(candidate)
                parsed = value if isinstance(value, dict) else {}
            except (TypeError, ValueError):
                parsed = {}
        explicit = find_value(upstream, "emotion_needs_support")
        if explicit is None:
            explicit = parsed.get("emotion_needs_support", payload.get("emotion_needs_support"))
        score = find_value(upstream, "emotional_distress_score")
        if score is None:
            score = parsed.get("emotional_distress_score", payload.get("emotional_distress_score"))
        threshold = float(config.get("parameters", {}).get("emotion_threshold", 0.6))
        needs_support = bool(explicit) if explicit is not None else float(score or 0) >= threshold
        return {
            "emotion_needs_support": needs_support,
            "selected_route": "emotion_support" if needs_support else "learning_practice",
            "emotional_distress_score": score,
            "emotion_evidence": parsed.get("emotion_evidence", find_value(upstream, "emotion_evidence") or []),
            "mastery_by_knowledge_point": find_value(upstream, "mastery_by_knowledge_point"),
            "overall_mastery": find_value(upstream, "overall_mastery"),
            "confidence": find_value(upstream, "confidence"),
            "target_mastery": find_value(upstream, "target_mastery") or payload.get("target_mastery", 0.8),
        }
    if operation == "validate_exercise_set":
        exercises = find_value(upstream, "exercises") or payload.get("exercises") or []
        required = ("question", "answer", "explanation", "knowledge_point")
        valid, issues, seen = [], [], set()
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
        return {"validated_exercises": valid, "quality": {"valid": bool(valid) and not issues, "issues": issues}}
    if operation == "score_learning_responses":
        exercises = find_value(upstream, "validated_exercises") or payload.get("exercises") or []
        responses = find_value(upstream, "learner_responses") or payload.get("learner_responses") or []
        answer_key = {str(item.get("id", i)): item for i, item in enumerate(exercises) if isinstance(item, dict)}
        item_scores, totals = [], {}
        for index, response in enumerate(responses if isinstance(responses, list) else []):
            if not isinstance(response, dict):
                continue
            item_id = str(response.get("exercise_id", response.get("id", index)))
            exercise = answer_key.get(item_id, {})
            score = float(
                bool(exercise.get("answer"))
                and str(response.get("answer", "")).strip().casefold()
                == str(exercise.get("answer", "")).strip().casefold()
            )
            knowledge = str(exercise.get("knowledge_point", "unknown"))
            totals.setdefault(knowledge, []).append(score)
            item_scores.append({"exercise_id": item_id, "knowledge_point": knowledge, "score": score})
        by_knowledge = {name: round(sum(values) / len(values), 4) for name, values in totals.items()}
        overall = round(sum(item["score"] for item in item_scores) / max(1, len(item_scores)), 4)
        return {"item_scores": item_scores, "score_by_knowledge_point": by_knowledge, "overall_score": overall}
    if operation == "evaluate_threshold":
        score = float(find_value(upstream, "overall_score") or payload.get("score", 0))
        threshold = float(payload.get("threshold", config.get("parameters", {}).get("threshold", 0.8)))
        return {"score": score, "threshold": threshold, "passed": score >= threshold}
    if operation == "validate_payload":
        required = config.get("parameters", {}).get("required_fields", [])
        missing = [field for field in required if str(field).split(".", 1)[0].replace("[]", "") not in payload]
        return {"valid": not missing, "missing_fields": missing, "content": payload}
    return {"payload": payload, "upstream": upstream, "operation": operation}


def resolve_path(value, path):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def parse_literal(value):
    stripped = value.strip().strip("\"'")
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() in {"none", "null"}:
        return None
    try:
        return float(stripped) if "." in stripped else int(stripped)
    except ValueError:
        return stripped


def condition_matches(expression, output, payload):
    if not expression or expression == "else":
        return False
    match = re.fullmatch(r"\s*([A-Za-z_][\w.]*)\s*(==|!=|>=|<=|>|<)?\s*(.*?)\s*", expression)
    if not match:
        return False
    path, operator, raw_expected = match.groups()
    actual = resolve_path(payload, path[8:]) if path.startswith("payload.") else resolve_path(output, path)
    if not operator:
        return bool(actual)
    expected = parse_literal(raw_expected)
    try:
        return {"==": actual == expected, "!=": actual != expected, ">=": actual >= expected,
                "<=": actual <= expected, ">": actual > expected, "<": actual < expected}[operator]
    except TypeError:
        return False


def select_targets(node_id, output, outgoing, loop_counts, payload):
    edges = outgoing.get(node_id, [])
    for edge in edges:
        if edge.get("edge_type") != "loop":
            continue
        key = f"{node_id}->{edge['target']}"
        maximum = int(edge.get("max_iterations", 1))
        if loop_counts.get(key, 0) < maximum and condition_matches(edge.get("condition"), output, payload):
            loop_counts[key] = loop_counts.get(key, 0) + 1
            return [edge["target"]]
    controlled = any(edge.get("edge_type") != "default" or edge.get("condition") for edge in edges)
    if controlled:
        for edge in edges:
            if edge.get("edge_type") == "conditional" and edge.get("condition") not in {None, "", "else"}:
                if condition_matches(edge["condition"], output, payload):
                    return [edge["target"]]
        fallback = next((edge for edge in edges if edge.get("edge_type", "default") == "default" or edge.get("condition") in {None, "", "else"}), None)
        return [fallback["target"]] if fallback else []
    return [edge["target"] for edge in edges]


def main():
    parser = argparse.ArgumentParser(description="Run exported Adaptive-AGES workflow")
    parser.add_argument("--input", default="input.json")
    args = parser.parse_args()
    plan = json.loads(Path("workflow.json").read_text(encoding="utf-8"))
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    outputs = {}
    nodes = {node["id"]: node for node in plan["nodes"]}
    predecessors = {node_id: [] for node_id in nodes}
    outgoing = {node_id: [] for node_id in nodes}
    for edge in plan["edges"]:
        outgoing[edge["source"]].append(edge)
        if edge.get("edge_type") != "loop":
            predecessors[edge["target"]].append(edge["source"])
    queue = [node_id for node_id, deps in predecessors.items() if not deps]
    if not queue:
        raise RuntimeError("工作流没有入口节点")
    loop_counts, executions = {}, 0
    while queue:
        node_id = queue.pop(0)
        node = nodes[node_id]
        upstream = {dep: outputs.get(dep) for dep in predecessors[node_id] if dep in outputs}
        kind = node["subject_type"]
        if kind == "llm":
            result = run_llm(node, payload, upstream)
        elif kind == "ml":
            result = run_ml(node, payload, upstream)
        elif kind == "human":
            config = node.get("config", {})
            print(f"\n需要人工处理：{node['label']}\n{config.get('instruction', '')}")
            print(f"验收标准：{config.get('approval_criteria', '')}")
            print(f"提交字段：{config.get('response_schema', {}).get('required_fields', [])}")
            human_input = input("请输入复核意见（输入 revise 可触发修订循环）： ")
            needs_revision = human_input.strip().lower() in {"revise", "revision", "修改", "需要修改", "不通过"}
            result = {"human_input": human_input, "needs_revision": needs_revision, "approved": not needs_revision}
        else:
            result = run_tool(node, payload, upstream)
        outputs[node_id] = result
        executions += 1
        if executions > 100:
            raise RuntimeError("工作流超过安全执行上限，请检查循环条件")
        for target in select_targets(node_id, result, outgoing, loop_counts, payload):
            is_loop = any(edge.get("target") == target and edge.get("edge_type") == "loop" for edge in outgoing[node_id])
            if is_loop or all(dep in outputs for dep in predecessors[target]):
                if target not in queue:
                    queue.append(target)
        print(f"完成：{node['label']}")
    Path("output.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果已写入 output.json")


if __name__ == "__main__":
    main()
'''


def _safe_node_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def export_workflow_bundle(plan: dict[str, Any], name: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workflow.json", json.dumps(plan, ensure_ascii=False, indent=2))
        archive.writestr("run_workflow.py", RUNNER)
        archive.writestr("input.json", json.dumps({"task_input": "请替换为实际输入", "features": []}, ensure_ascii=False, indent=2))
        archive.writestr("requirements.txt", "openai>=1.60\npython-dotenv>=1.0\nnumpy>=2.0\nscikit-learn>=1.6\n")
        archive.writestr(".env.example", "OPENAI_API_KEY=\nOPENAI_BASE_URL=\nOPENAI_MODEL=gpt-4.1-mini\n")
        archive.writestr(
            "README.md",
            f"# {name}\n\n这是 Adaptive-AGES 导出的独立可运行工作流。\n\n"
            "1. 创建虚拟环境：`python -m venv .venv`\n"
            "2. 激活后安装：`pip install -r requirements.txt`\n"
            "3. 设置环境变量中的模型 API（可复制 `.env.example`）\n"
            "4. 修改 `input.json`\n"
            "5. 执行：`python run_workflow.py --input input.json`\n\n"
            "ML 节点位于 `nodes/`，每个文件必须实现 `run(payload, upstream)`。\n",
        )
        for node in plan.get("nodes", []):
            if node.get("subject_type") == "ml":
                code = node.get("config", {}).get("code") or "def run(payload, upstream):\n    return {'payload': payload}\n"
                archive.writestr(f"nodes/{_safe_node_id(node['id'])}.py", code)
    return buffer.getvalue()
