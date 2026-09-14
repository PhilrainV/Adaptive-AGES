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
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=float(config.get("temperature", 0.2)),
        messages=[
            {"role": "system", "content": config.get("system_prompt", "你是严谨的任务执行智能体。")},
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


def main():
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
