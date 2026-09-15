"""Agent 2: generate psychometrically annotated, task-grounded diagnostic items.

Edit ``DEFAULT_SYSTEM_PROMPT``, ``DEFAULT_SKILLS`` or ``ASSESSMENT_BLUEPRINT`` to
upgrade the agent without changing the API orchestration layer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas.domain import AgentRuntimeConfig, AgentSkill, TaskGraph
from app.services.human_assessment import HumanCapabilityAssessmentService

Dimension = Literal[
    "programming",
    "ai_literacy",
    "domain_knowledge",
    "workflow_design",
    "judgement",
]

ASSESSMENT_BLUEPRINT: dict[str, int] = {
    "programming": 2,
    "ai_literacy": 2,
    "domain_knowledge": 2,
    "workflow_design": 2,
    "judgement": 2,
}

DEFAULT_SYSTEM_PROMPT = """你是 Adaptive-AGES 的认知诊断出题 Agent。根据任务图生成 10 道单选情境题，
测量用户在本次任务中亲自参与所需的五种属性，而不是测量 LLM 或 ML 的能力。

每道题必须：
1. 与一个具体子任务及真实决策情境关联；
2. 指定 dimension 和 knowledge_components，后者构成 DINA 模型的 Q 矩阵；
3. 提供 4 个有诊断价值的选项，只有一个最佳答案；
4. 给出 difficulty、discrimination、guess、slip 的初始标定值；
5. 避免纯记忆题，优先测量证据判断、操作选择和错误识别。

五个属性各至少有两题，且每个属性至少有一道单属性锚题，以保证 Q 矩阵可识别。guess 通常为 0.15-0.30，
slip 通常为 0.05-0.25。题目将在后端用贝叶斯 DINA 估计属性掌握概率。
"""

DEFAULT_SKILLS = [
    AgentSkill(name="q_matrix_design", description="构造可识别 Q 矩阵", instructions="每个属性至少一题单属性锚题，并平衡覆盖。"),
    AgentSkill(name="scenario_item_writing", description="任务情境题设计", instructions="把抽象能力放进用户给出的真实子任务。"),
    AgentSkill(name="distractor_diagnosis", description="诊断性干扰项", instructions="干扰项对应常见误解、遗漏或不当决策。"),
    AgentSkill(name="item_calibration", description="题目参数初始标定", instructions="为 DINA 提供保守的 guess/slip 初值。"),
]


class GeneratedQuestion(BaseModel):
    id: str
    dimension: Dimension
    knowledge_components: list[Dimension] = Field(min_length=1, max_length=3)
    subtask_id: str | None = None
    prompt: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    rationale: str = ""
    difficulty: float = Field(default=.5, ge=0, le=1)
    discrimination: float = Field(default=1, ge=.25, le=2.5)
    guess: float = Field(default=.22, ge=.05, le=.4)
    slip: float = Field(default=.12, ge=.02, le=.4)


class GeneratedAssessment(BaseModel):
    questions: list[GeneratedQuestion] = Field(min_length=8, max_length=12)


class TestGenerationAgent:
    """Task-grounded item generator with Q-matrix quality validation."""

    __test__ = False
    name = "test_generation_agent"
    prompt_version = "2.0"

    def __init__(
        self,
        fallback: HumanCapabilityAssessmentService | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        skills: list[AgentSkill] | None = None,
    ):
        self.fallback = fallback or HumanCapabilityAssessmentService()
        self.system_prompt = system_prompt
        self.skills = skills or list(DEFAULT_SKILLS)

    async def run(
        self,
        task_graph: TaskGraph,
        model_config: dict | None = None,
        agent_config: AgentRuntimeConfig | dict[str, Any] | None = None,
    ) -> tuple[list[dict], str]:
        config = self._normalise_config(agent_config)
        system_prompt = config.get("system_prompt") or self.system_prompt
        skills = self.skills + self._skills_from_config(config)
        if not model_config or not model_config.get("api_key"):
            questions = self.fallback.generate(task_graph.goal, task_graph)
            self.validate_blueprint(questions)
            return questions, "dina_rule"

        task_summary = "\n".join(
            f"- {item.id} | {item.name}: {item.description}; "
            f"preferred={[kind.value for kind in item.preferred_subject_types]}; risk={item.risk:.2f}"
            for item in task_graph.subtasks
        )
        skill_text = "\n".join(
            f"- {skill.name}: {skill.description}。{skill.instructions}"
            for skill in skills
            if skill.enabled
        )
        try:
            llm = ChatOpenAI(
                api_key=model_config["api_key"],
                base_url=model_config.get("base_url") or None,
                model=model_config.get("model") or "gpt-4.1-mini",
                temperature=.15,
                timeout=60,
            ).with_structured_output(GeneratedAssessment)
            generated = await llm.ainvoke(
                f"{system_prompt}\n\n可用出题技能：\n{skill_text}\n\n"
                f"用户问题：{task_graph.goal}\n任务与主体分析：\n{task_summary}"
            )
            questions = self._clean(generated.questions, task_graph)
            self.validate_blueprint(questions)
            return questions, "llm_dina"
        except Exception:  # noqa: BLE001 - provider or psychometric validation fallback
            questions = self.fallback.generate(task_graph.goal, task_graph)
            self.validate_blueprint(questions)
            return questions, "dina_rule_fallback"

    @staticmethod
    def _clean(questions: list[GeneratedQuestion], task_graph: TaskGraph) -> list[dict]:
        valid_subtasks = {item.id for item in task_graph.subtasks}
        seen_ids: set[str] = set()
        cleaned: list[dict] = []
        for index, question in enumerate(questions[:10]):
            item = question.model_dump(mode="json")
            item_id = item["id"].strip() or f"q{index + 1}"
            if item_id in seen_ids:
                item_id = f"q{index + 1}"
            components = list(dict.fromkeys(item["knowledge_components"]))
            if item["dimension"] not in components:
                components.insert(0, item["dimension"])
            item.update(
                id=item_id,
                prompt=item["prompt"].strip(),
                options=[str(option).strip() for option in item["options"]],
                knowledge_components=components[:3],
                subtask_id=item["subtask_id"] if item["subtask_id"] in valid_subtasks else None,
            )
            seen_ids.add(item_id)
            cleaned.append(item)
        return cleaned

    @staticmethod
    def validate_blueprint(questions: list[dict]) -> None:
        if len(questions) < 8:
            raise ValueError("diagnostic assessment needs at least eight items")
        coverage: Counter[str] = Counter()
        anchors: Counter[str] = Counter()
        for item in questions:
            components = item.get("knowledge_components") or [item["dimension"]]
            coverage.update(components)
            if len(components) == 1:
                anchors[components[0]] += 1
            if len(item.get("options", [])) != 4:
                raise ValueError("every diagnostic item must have exactly four options")
        missing = [dimension for dimension in ASSESSMENT_BLUEPRINT if coverage[dimension] < 2]
        missing_anchors = [dimension for dimension in ASSESSMENT_BLUEPRINT if anchors[dimension] < 1]
        if missing or missing_anchors:
            raise ValueError(f"invalid Q-matrix coverage={missing}, anchors={missing_anchors}")

    @staticmethod
    def _normalise_config(config: AgentRuntimeConfig | dict[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        return config.model_dump() if isinstance(config, AgentRuntimeConfig) else dict(config)

    @staticmethod
    def _skills_from_config(config: dict[str, Any]) -> list[AgentSkill]:
        return [skill if isinstance(skill, AgentSkill) else AgentSkill.model_validate(skill) for skill in config.get("skills", [])]
