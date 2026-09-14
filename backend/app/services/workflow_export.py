import io
import json
import re
import zipfile
from copy import deepcopy
from typing import Any

RUNNER = r'''import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv


def render(template, payload, upstream):
    return template.replace("{input}", json.dumps(payload, ensure_ascii=False)).replace("{upstream}", json.dumps(upstream, ensure_ascii=False))


def run_llm(node, payload, upstream):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt") from exc
    config = node.get("config", {})
    use_default = config.get("use_default_model", True)
    key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get("OPENAI_API_KEY") if use_default else os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"节点 {node['label']} 缺少环境变量 {key_env}")
    base_url = os.environ.get("OPENAI_BASE_URL") if use_default else config.get("base_url")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini") if use_default else config.get("model")
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    prompt = render(config.get("prompt_template", "{input}\n{upstream}"), payload, upstream)
    user_content = prompt
    if config.get("modality") == "vision":
        urls = list(config.get("image_urls") or [])
        if isinstance(payload.get("image_url"), str):
            urls.append(payload["image_url"])
        if isinstance(payload.get("image_urls"), list):
            urls.extend(str(url) for url in payload["image_urls"] if url)
        if urls:
            user_content = [{"type": "text", "text": prompt}]
            user_content.extend({"type": "image_url", "image_url": {"url": url}} for url in urls)
    response = client.chat.completions.create(
        model=model,
        temperature=float(config.get("temperature", 0.2)),
        messages=[
            {"role": "system", "content": config.get("system_prompt", "你是严谨的任务执行智能体。")},
            {"role": "user", "content": user_content},
        ],
    )
    return {"content": response.choices[0].message.content, "model": model}


def run_ml(node, payload, upstream):
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", node["id"])
    path = Path("nodes") / f"{safe_id}.py"
    spec = importlib.util.spec_from_file_location(node["id"].replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(payload, upstream)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run exported Adaptive-AGES workflow")
    parser.add_argument("--input", default="input.json")
    args = parser.parse_args()
    plan = json.loads(Path("workflow.json").read_text(encoding="utf-8"))
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    outputs, pending = {}, {node["id"]: node for node in plan["nodes"]}
    predecessors = {node_id: [] for node_id in pending}
    for edge in plan["edges"]:
        predecessors[edge["target"]].append(edge["source"])
    while pending:
        ready = [node for node in pending.values() if all(dep in outputs for dep in predecessors[node["id"]])]
        if not ready:
            raise RuntimeError("工作流存在环或依赖缺失")
        for node in ready:
            upstream = {dep: outputs[dep] for dep in predecessors[node["id"]]}
            kind = node["subject_type"]
            if kind == "llm":
                result = run_llm(node, payload, upstream)
            elif kind == "ml":
                result = run_ml(node, payload, upstream)
            elif kind == "human":
                print(f"\n需要人工处理：{node['label']}\n{node.get('config', {}).get('instruction', '')}")
                result = {"human_input": input("请输入复核意见： ")}
            else:
                result = {"payload": payload, "upstream": upstream}
            outputs[node["id"]] = result
            del pending[node["id"]]
            print(f"完成：{node['label']}")
    Path("output.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果已写入 output.json")


if __name__ == "__main__":
    main()
'''


def _safe_node_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _export_plan(plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    exported = deepcopy(plan)
    node_envs: list[str] = []
    for node in exported.get("nodes", []):
        config = node.get("config") or {}
        encrypted = config.pop("api_key_encrypted", None)
        configured = bool(encrypted or config.pop("api_key_configured", False))
        if node.get("subject_type") == "llm" and not config.get("use_default_model", True) and configured:
            env_name = f"NODE_{_safe_node_id(node['id']).upper().replace('-', '_')}_API_KEY"
            config["api_key_env"] = env_name
            node_envs.append(env_name)
        node["config"] = config
    return exported, node_envs


def export_workflow_bundle(plan: dict[str, Any], name: str) -> bytes:
    exported_plan, node_envs = _export_plan(plan)
    env_lines = ["OPENAI_API_KEY=", "OPENAI_BASE_URL=", "OPENAI_MODEL=gpt-4.1-mini"]
    env_lines.extend(f"{name}=" for name in node_envs)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workflow.json", json.dumps(exported_plan, ensure_ascii=False, indent=2))
        archive.writestr("run_workflow.py", RUNNER)
        archive.writestr("input.json", json.dumps({"task_input": "请替换为实际输入", "features": [], "image_urls": []}, ensure_ascii=False, indent=2))
        archive.writestr("requirements.txt", "openai>=1.60\npython-dotenv>=1.0\nnumpy>=2.0\nscikit-learn>=1.6\n")
        archive.writestr(".env.example", "\n".join(env_lines) + "\n")
        archive.writestr(
            "README.md",
            f"# {name}\n\n这是 Adaptive-AGES 导出的独立可运行工作流。\n\n"
            "1. 创建虚拟环境：`python -m venv .venv`\n"
            "2. 激活后安装：`pip install -r requirements.txt`\n"
            "3. 复制 `.env.example` 为 `.env`，填写默认模型及各节点 API Key\n"
            "4. 修改 `input.json`；视觉节点可填写 image_url 或 image_urls\n"
            "5. 执行：`python run_workflow.py --input input.json`\n\n"
            "独立 LLM 节点使用各自的 NODE_*_API_KEY；密钥不会写入导出包。\n"
            "ML 节点位于 `nodes/`，每个文件必须实现 `run(payload, upstream)`。\n",
        )
        for node in exported_plan.get("nodes", []):
            if node.get("subject_type") == "ml":
                code = node.get("config", {}).get("code") or "def run(payload, upstream):\n    return {'payload': payload}\n"
                archive.writestr(f"nodes/{_safe_node_id(node['id'])}.py", code)
    return buffer.getvalue()
