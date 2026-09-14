from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas.domain import TaskGraph
from app.services.human_assessment import HumanCapabilityAssessmentService


class GeneratedQuestion(BaseModel):
    id: str
    dimension: Literal[
        "programming", "ai_literacy", "domain_knowledge", "workflow_design", "judgement"
    ]
    prompt: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    rationale: str = ""


class GeneratedAssessment(BaseModel):
    questions: list[GeneratedQuestion] = Field(min_length=6, max_length=10)


class TestGenerationAgent:
    """Agent 2: creates a task-grounded test after the problem has been analysed."""

    name = "test_generation_agent"

    def __init__(self, fallback: HumanCapabilityAssessmentService | None = None):
        self.fallback = fallback or HumanCapabilityAssessmentService()

    async def run(
        self,
        task_graph: TaskGraph,
        model_config: dict | None = None,
    ) -> tuple[list[dict], str]:
        if not model_config or not model_config.get("api_key"):
            return self.fallback.generate(task_graph.goal), "rule"

        task_summary = "\n".join(
            f"- {item.name}: {item.description}; risk={item.risk:.2f}"
            for item in task_graph.subtasks
        )
        try:
            llm = ChatOpenAI(
                api_key=model_config["api_key"],
                base_url=model_config.get("base_url") or None,
                model=model_config.get("model") or "gpt-4.1-mini",
                temperature=0.2,
                timeout=45,
            ).with_structured_output(GeneratedAssessment)
            generated = await llm.ainvoke(
                "你是 Adaptive-AGES 的能力测试出题 Agent。根据用户问题及其任务分解，"
                "生成 8 道单选情境题，用于判断用户亲自参与该任务的能力。题目必须与当前任务"
                "紧密相关，而不是通用 AI 常识题。覆盖 programming、ai_literacy、"
                "domain_knowledge、workflow_design、judgement 五个维度；每题恰好 4 个选项，"
                "只有一个最佳答案。不要考死记硬背，要考任务决策、证据判断和实际操作。\n\n"
                f"用户问题：{task_graph.goal}\n任务分解：\n{task_summary}"
            )
            questions = []
            seen_ids: set[str] = set()
            for index, question in enumerate(generated.questions[:8]):
                item = question.model_dump(mode="json")
                item_id = item["id"].strip() or f"q{index + 1}"
                if item_id in seen_ids:
                    item_id = f"q{index + 1}"
                item["id"] = item_id
                item["prompt"] = item["prompt"].strip()
                item["options"] = [str(option).strip() for option in item["options"]]
                seen_ids.add(item_id)
                questions.append(item)
            covered = {item["dimension"] for item in questions}
            if len(questions) < 6 or len(covered) < 5:
                raise ValueError("generated assessment lacks dimension coverage")
            return questions, "llm"
        except Exception:  # noqa: BLE001 - providers may not support structured output
            return self.fallback.generate(task_graph.goal), "rule_fallback"
