"""Agent 1: analyse a problem before any assessment or workflow is created.

Research extension points are intentionally kept in this file:
1. ``DEFAULT_SYSTEM_PROMPT`` controls the model's analytical frame.
2. ``DEFAULT_SKILLS`` can be extended with domain-specific analysis skills.
3. ``_enrich_fallback`` defines the auditable non-LLM policy.
4. ``_clean_subtasks`` is the structural validity gate for model output.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas.domain import (
    AgentRuntimeConfig,
    AgentSkill,
    ExecutionMode,
    IterationPolicy,
    SubjectSuitability,
    SubjectType,
    Subtask,
    TaskGraph,
)
from app.services.task_understanding import TaskUnderstandingEngine

DEFAULT_SYSTEM_PROMPT = """你是 Adaptive-AGES 的问题解析 Agent。你的职责不是直接生成工作流，而是把
用户目标转换为可供后续测评与优化算法使用的任务图。

对每个子任务必须同时完成四类分析：
1. 描述能力需求、风险和数据/输出契约；
2. 分别判断 LLM、ML、Human、Tool 的适用程度、角色和理由，不能只给一个标签；
3. 给出首选与不适合的主体类型，使规划器能够计算替代方案；
4. 判断控制流是顺序、并行、条件还是迭代，并在确有证据时给出条件或反馈回路。

当需求中包含多个明确动作、阶段、条件或“完成后/直到”等时序关系时，必须拆成 3—10 个可独立分配和执行的
子任务，不能把“分析—生成—执行—检查—调整—总结”压缩为一个生成节点。学习者作答、人工填写、教师确认等
真实人类活动使用 human_action 或 human_review；规则检查、状态判断和接口操作可使用 evaluation 或 tool。

分配原则：LLM 擅长语义推理、生成和非结构化信息处理；ML 擅长稳定的结构化预测、分类和数值计算；
Human 擅长价值判断、情境知识、责任确认和高风险复核；Tool 擅长确定性接口、检索、转换与执行。
不要为了图复杂而强行增加分支或循环。只有存在不确定性阈值、审核不通过、缺失数据、反馈修订等真实条件时
才使用条件或迭代。依赖必须指向更早出现的子任务；循环只能通过 iteration_policy 回指较早子任务。
"""

DEFAULT_SKILLS = [
    AgentSkill(
        name="heterogeneous_subject_analysis",
        description="比较 LLM、ML、人类与工具的边界",
        instructions="对每个子任务给出四类主体适配证据与替代关系。",
    ),
    AgentSkill(
        name="control_flow_design",
        description="识别真实的并行、条件与反馈关系",
        instructions="将可同时执行的工作并行化；将低置信度、审核失败和缺失数据表示成条件或循环。",
    ),
    AgentSkill(
        name="risk_and_accountability",
        description="识别必须保留人类责任的节点",
        instructions="涉及权益、伦理、不可逆操作或专业责任时，提高 Human 的适配度并说明责任。",
    ),
]


class GeneratedBreakdown(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1, max_length=10)
    assignment_summary: dict[str, list[str]] = Field(default_factory=dict)


class ProblemAnalysisAgent:
    """Prompt-driven task analyst with explicit skill and policy extension points."""

    name = "problem_analysis_agent"
    prompt_version = "2.0"

    def __init__(
        self,
        engine: TaskUnderstandingEngine | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        skills: list[AgentSkill] | None = None,
    ):
        self.engine = engine or TaskUnderstandingEngine()
        self.system_prompt = system_prompt
        self.skills = skills or list(DEFAULT_SKILLS)

    async def run(
        self,
        prompt: str,
        model_config: dict | None = None,
        agent_config: AgentRuntimeConfig | dict[str, Any] | None = None,
    ) -> TaskGraph:
        config = self._normalise_config(agent_config)
        system_prompt = config.get("system_prompt") or self.system_prompt
        skills = self.skills + self._skills_from_config(config)
        if not model_config or not model_config.get("api_key"):
            return self._enrich_fallback(self.engine.understand(prompt), skills)

        try:
            llm = ChatOpenAI(
                api_key=model_config["api_key"],
                base_url=model_config.get("base_url") or None,
                model=model_config.get("model") or "gpt-4.1-mini",
                temperature=0,
                timeout=60,
            ).with_structured_output(GeneratedBreakdown)
            result = await llm.ainvoke(
                f"{system_prompt}\n\n可用分析技能：\n{self._render_skills(skills)}\n\n用户需求：{prompt}"
            )
            subtasks = self._clean_subtasks(result.subtasks)
            if not subtasks:
                raise ValueError("problem analysis returned no valid subtasks")
            if not self._coverage_is_sufficient(prompt, subtasks):
                raise ValueError("multi-stage requirement was under-decomposed")
            graph = TaskGraph(
                task_id=hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
                goal=prompt,
                complexity=self._complexity(subtasks),
                subtasks=subtasks,
                planning_mode="llm_subject_aware",
                assignment_summary=result.assignment_summary or self._assignment_summary(subtasks),
                analysis_trace=self._analysis_trace(subtasks, skills, "llm"),
            )
            return graph
        except Exception as exc:  # noqa: BLE001 - provider/structured-output fallback
            graph = self._enrich_fallback(self.engine.understand(prompt), skills)
            graph.planning_mode = "rule_fallback"
            graph.analysis_trace.append({"fallback_reason": type(exc).__name__})
            return graph

    def _enrich_fallback(self, graph: TaskGraph, skills: list[AgentSkill]) -> TaskGraph:
        enriched: list[Subtask] = []
        for item in graph.subtasks:
            fits = self._default_suitability(item.task_type, item.risk)
            preferred = [fit.subject_type for fit in fits if fit.suitability >= .72]
            unsuitable = [fit.subject_type for fit in fits if fit.suitability <= .25]
            iteration = item.iteration_policy
            if item.task_type == "human_review" and enriched:
                target = next(
                    (candidate.id for candidate in reversed(enriched) if candidate.task_type in {"generation", "reasoning"}),
                    None,
                )
                if target:
                    iteration = IterationPolicy(
                        enabled=True,
                        feedback_target_subtask_id=target,
                        condition="needs_revision == true",
                        max_iterations=2,
                    )
            enriched.append(
                item.model_copy(
                    update={
                        "subject_suitability": fits,
                        "preferred_subject_types": preferred,
                        "unsuitable_subject_types": unsuitable,
                        "assignment_rationale": self._assignment_rationale(fits),
                        "execution_mode": ExecutionMode.ITERATIVE if iteration.enabled else item.execution_mode,
                        "iteration_policy": iteration,
                    }
                )
            )
        self._mark_parallel_groups(enriched)
        return graph.model_copy(
            update={
                "subtasks": enriched,
                "assignment_summary": self._assignment_summary(enriched),
                "analysis_trace": self._analysis_trace(enriched, skills, "rule"),
            }
        )

    @staticmethod
    def _clean_subtasks(items: list[Subtask]) -> list[Subtask]:
        seen: set[str] = set()
        cleaned: list[Subtask] = []
        for index, item in enumerate(items[:10]):
            item_id = item.id.strip() or f"step-{index + 1}"
            if item_id in seen:
                item_id = f"{item_id}-{index + 1}"
            dependencies = [dep for dep in item.dependencies if dep in seen]
            policy = item.iteration_policy
            if policy.enabled and policy.feedback_target_subtask_id not in seen:
                policy = IterationPolicy()
            fits = sorted(item.subject_suitability, key=lambda fit: fit.suitability, reverse=True)
            preferred = item.preferred_subject_types or [fit.subject_type for fit in fits if fit.suitability >= .72]
            cleaned.append(
                item.model_copy(
                    update={
                        "id": item_id,
                        "dependencies": dependencies,
                        "subject_suitability": fits,
                        "preferred_subject_types": preferred,
                        "iteration_policy": policy,
                    }
                )
            )
            seen.add(item_id)
        ProblemAnalysisAgent._mark_parallel_groups(cleaned)
        return cleaned

    @staticmethod
    def _mark_parallel_groups(subtasks: list[Subtask]) -> None:
        groups: dict[tuple[str, ...], list[Subtask]] = defaultdict(list)
        for item in subtasks:
            if item.dependencies:
                groups[tuple(sorted(item.dependencies))].append(item)
        for siblings in groups.values():
            if len(siblings) > 1:
                for item in siblings:
                    if item.execution_mode == ExecutionMode.SEQUENTIAL:
                        item.execution_mode = ExecutionMode.PARALLEL

    @staticmethod
    def _default_suitability(task_type: str, risk: float) -> list[SubjectSuitability]:
        matrix = {
            "data_processing": {SubjectType.TOOL: .95, SubjectType.ML: .86, SubjectType.LLM: .38, SubjectType.HUMAN: .35},
            "prediction": {SubjectType.ML: .97, SubjectType.LLM: .42, SubjectType.HUMAN: .44, SubjectType.TOOL: .52},
            "generation": {SubjectType.LLM: .96, SubjectType.HUMAN: .67, SubjectType.ML: .28, SubjectType.TOOL: .25},
            "reasoning": {SubjectType.LLM: .91, SubjectType.HUMAN: .76, SubjectType.ML: .34, SubjectType.TOOL: .25},
            "human_review": {SubjectType.HUMAN: .99, SubjectType.LLM: .46, SubjectType.ML: .22, SubjectType.TOOL: .18},
            "human_action": {SubjectType.HUMAN: .99, SubjectType.LLM: .12, SubjectType.ML: .08, SubjectType.TOOL: .12},
            "evaluation": {SubjectType.TOOL: .9, SubjectType.ML: .82, SubjectType.HUMAN: .7, SubjectType.LLM: .62},
            "tool": {SubjectType.TOOL: .98, SubjectType.LLM: .35, SubjectType.ML: .3, SubjectType.HUMAN: .25},
        }
        values = matrix.get(task_type, matrix["reasoning"])
        if risk >= .75:
            values = {**values, SubjectType.HUMAN: max(values[SubjectType.HUMAN], .9)}
        roles = {SubjectType.HUMAN: "judge", SubjectType.LLM: "semantic_executor", SubjectType.ML: "predictor", SubjectType.TOOL: "deterministic_executor"}
        return [
            SubjectSuitability(subject_type=kind, suitability=score, role=roles[kind], rationale=f"{task_type} 对 {kind.value} 的规则适配度为 {score:.0%}")
            for kind, score in sorted(values.items(), key=lambda pair: pair[1], reverse=True)
        ]

    @staticmethod
    def _assignment_rationale(fits: list[SubjectSuitability]) -> str:
        ordered = sorted(fits, key=lambda fit: fit.suitability, reverse=True)
        return "；".join(f"{fit.subject_type.value}:{fit.suitability:.0%}" for fit in ordered)

    @staticmethod
    def _assignment_summary(subtasks: list[Subtask]) -> dict[str, list[str]]:
        summary = {kind.value: [] for kind in SubjectType}
        for item in subtasks:
            kinds = item.preferred_subject_types or [fit.subject_type for fit in item.subject_suitability[:1]]
            for kind in kinds:
                summary[kind.value].append(item.id)
        return summary

    @staticmethod
    def _complexity(subtasks: list[Subtask]) -> float:
        breadth = sum(
            value >= .7
            for item in subtasks
            for value in item.requirement.model_dump().values()
        )
        control_bonus = sum(item.execution_mode != ExecutionMode.SEQUENTIAL for item in subtasks)
        return round(min(.98, .16 + len(subtasks) * .11 + breadth * .025 + control_bonus * .04), 3)

    @staticmethod
    def _requires_decomposition(prompt: str) -> bool:
        text = prompt.lower()
        action_groups = [
            ["分析", "诊断", "识别", "assess", "analyse"],
            ["预测", "分类", "predict", "forecast"],
            ["生成", "制定", "create", "generate"],
            ["完成", "作答", "填写", "execute"],
            ["检查", "评价", "验证", "check", "evaluate"],
            ["调整", "修订", "重新", "迭代", "循环", "revise", "iterate"],
            ["总结", "报告", "summary", "report"],
        ]
        action_count = sum(any(marker in text for marker in group) for group in action_groups)
        control_flow = any(marker in text for marker in ["如果", "完成后", "仍然", "直到", "否则", "if ", "until", "after"])
        return action_count >= 3 or (action_count >= 2 and control_flow)

    @staticmethod
    def _coverage_is_sufficient(prompt: str, subtasks: list[Subtask]) -> bool:
        if ProblemAnalysisAgent._requires_decomposition(prompt) and len(subtasks) < 3:
            return False
        text = prompt.lower()
        if any(marker in text for marker in ["如果", "否则", "if "]) and not any(
            item.entry_condition for item in subtasks
        ):
            return False
        if any(marker in text for marker in ["直到", "循环", "迭代", "重新生成", "继续调整", "until", "iterate"]) and not any(
            item.iteration_policy.enabled for item in subtasks
        ):
            return False
        explicit_human_action = any(
            marker in text
            for marker in ["完成每个练习", "完成练习", "学生完成", "用户完成", "作答", "人工填写"]
        )
        if explicit_human_action and not any(
            item.task_type in {"human_action", "human_review"}
            or SubjectType.HUMAN in item.preferred_subject_types
            for item in subtasks
        ):
            return False
        practice_and_summary = any(marker in text for marker in ["练习", "习题", "训练题"]) and any(
            marker in text for marker in ["总结", "最终报告", "summary", "report"]
        )
        return not (
            practice_and_summary
            and sum(item.task_type == "generation" for item in subtasks) < 2
        )

    def _analysis_trace(self, subtasks: list[Subtask], skills: list[AgentSkill], mode: str) -> list[dict[str, Any]]:
        return [{
            "agent": self.name,
            "prompt_version": self.prompt_version,
            "mode": mode,
            "skills": [skill.name for skill in skills if skill.enabled],
            "subject_analysis_complete": all(bool(item.subject_suitability) for item in subtasks),
            "control_flow": {
                mode.value: sum(item.execution_mode == mode for item in subtasks)
                for mode in ExecutionMode
            },
        }]

    @staticmethod
    def _normalise_config(config: AgentRuntimeConfig | dict[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        return config.model_dump() if isinstance(config, AgentRuntimeConfig) else dict(config)

    @staticmethod
    def _skills_from_config(config: dict[str, Any]) -> list[AgentSkill]:
        return [skill if isinstance(skill, AgentSkill) else AgentSkill.model_validate(skill) for skill in config.get("skills", [])]

    @staticmethod
    def _render_skills(skills: list[AgentSkill]) -> str:
        return "\n".join(
            f"- {skill.name}: {skill.description}。{skill.instructions}"
            for skill in skills
            if skill.enabled
        )
