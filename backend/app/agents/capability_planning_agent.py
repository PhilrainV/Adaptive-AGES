"""Agent 4: comfort-aware global allocation and non-linear workflow synthesis.

The optimisation policy, objective weights and node templates live in this file so
researchers can replace individual components without touching API routes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from math import exp
from typing import Any
from uuid import uuid4

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas.domain import (
    AgentRuntimeConfig,
    AgentSkill,
    CapabilitySubject,
    EdgeType,
    ExecutionMode,
    SubjectType,
    Subtask,
    TaskGraph,
    WorkflowEdge,
    WorkflowNode,
    WorkflowPlan,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "fit": .42,
    "reliability": .15,
    "comfort": .18,
    "complementarity": .12,
    "cost": .08,
    "latency": .05,
}

DEFAULT_POLICY = {
    "beam_width": 64,
    "max_candidates_per_task": 4,
    "comfort_target_gap": .05,
    "comfort_bandwidth": .24,
    "human_overload_limit": .22,
    "load_balance_penalty": .045,
    "type_diversity_bonus": .015,
    "human_review_risk": .75,
}

DEFAULT_SYSTEM_PROMPT = """你是 Adaptive-AGES 的能力协同规划 Agent。你不是直接输出一个看似合理的线性流程，
而是根据任务图、用户认知诊断和可用的 LLM/ML/Human/Tool 主体，提出多个有实质差异的候选工作流。

每个候选方案必须：
1. 覆盖任务图中的全部子任务，并且只使用给定的 subtask_id 和 subject_id；
2. 明确每个子任务由哪个主体负责以及选择理由；
3. 根据真实依赖识别可并行部分，并考虑低置信度、数据缺失、审核不通过、执行失败等条件分支；
4. 对需要反馈修订的任务给出有界循环；对高风险或不可逆操作保留人类确认；
5. 体现不同策略取向，例如质量优先、低成本、低时延或人类舒适区优先，而不是生成内容相同的候选；
6. 不虚构任务图之外的能力、模型或工具，不用增加无意义复杂度来伪装智能。

候选方案只是提案，后续优化器会计算适配度、舒适区、可靠性、成本和时延，Critic 还会检查并修复约束。
"""

DEFAULT_CRITIC_PROMPT = """你是 Adaptive-AGES 的工作流 Critic。请独立审查每个候选工作流，而不是复述方案。
重点检查：任务覆盖、主体能力与用户舒适区、并行机会、条件分支、失败回退、有界迭代、高风险人类确认、
成本和时延。只报告可以定位和修复的问题；严重问题标为 error，一般改进标为 warning。
"""

DEFAULT_SKILLS = [
    AgentSkill(
        name="global_assignment_optimisation",
        description="在全部子任务上联合分配主体",
        instructions="避免逐节点贪心，比较多套完整分配方案的总体效用。",
    ),
    AgentSkill(
        name="human_comfort_zone_protection",
        description="保护用户能力舒适区",
        instructions="人的任务难度应略高于但接近其掌握水平，过大能力缺口由机器提供支架或接管。",
    ),
    AgentSkill(
        name="machine_complementarity",
        description="利用机器能力弥补人的薄弱维度",
        instructions="优先让 LLM、ML 或 Tool 覆盖认知诊断中证据充分的能力缺口。",
    ),
    AgentSkill(
        name="risk_governance",
        description="处理风险、确认与责任边界",
        instructions="高风险、价值判断和不可逆操作必须设置人类确认或明确的失败回退。",
    ),
    AgentSkill(
        name="conditional_and_iterative_flow_synthesis",
        description="构造必要的分支、并行与反馈循环",
        instructions="只在存在真实触发条件时创建条件或有界循环，同时识别可安全并行的任务。",
    ),
]


class ProposedAssignment(BaseModel):
    subtask_id: str
    subject_id: str
    role: str = "executor"
    rationale: str = ""


class ProposedFlowEdge(BaseModel):
    source_subtask_id: str
    target_subtask_id: str
    edge_type: EdgeType = EdgeType.DEFAULT
    condition: str | None = None
    max_iterations: int = Field(default=1, ge=1, le=10)


class CandidateProposal(BaseModel):
    name: str
    strategy: str
    assignments: list[ProposedAssignment]
    edges: list[ProposedFlowEdge] = Field(default_factory=list)
    human_review_subtask_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class CandidateProposalSet(BaseModel):
    candidates: list[CandidateProposal] = Field(min_length=2, max_length=6)


class CriticIssue(BaseModel):
    code: str
    severity: str = "warning"
    subtask_id: str | None = None
    description: str
    suggested_action: str = ""


class CandidateCritique(BaseModel):
    candidate_name: str
    accepted: bool = False
    issues: list[CriticIssue] = Field(default_factory=list)
    summary: str = ""


class CandidateCritiqueSet(BaseModel):
    critiques: list[CandidateCritique]


@dataclass(slots=True)
class AssignmentCandidate:
    subtask: Subtask
    subject: CapabilitySubject
    score: float
    fit: float
    type_prior: float
    comfort: float
    human_gap: float
    complementarity: float
    overload: float
    explanation: str


@dataclass(slots=True)
class BeamState:
    assignments: list[AssignmentCandidate] = field(default_factory=list)
    loads: dict[str, int] = field(default_factory=dict)
    type_loads: dict[str, int] = field(default_factory=dict)
    utility: float = 0.0


@dataclass(slots=True)
class PlanningCandidate:
    name: str
    strategy: str
    assignments: list[AssignmentCandidate]
    flow_edges: list[ProposedFlowEdge] = field(default_factory=list)
    human_review_subtasks: set[str] = field(default_factory=set)
    rationale: str = ""
    source: str = "algorithm"
    issues: list[dict[str, Any]] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    utility: float = 0.0


class CapabilityPlanningAgent:
    """Generate, optimise, critique, repair and select a capability-aware plan."""

    name = "capability_planning_agent"
    algorithm_version = "hybrid-deliberative-3.0"
    prompt_version = "3.0"

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        critic_prompt: str = DEFAULT_CRITIC_PROMPT,
        skills: list[AgentSkill] | None = None,
    ):
        self.system_prompt = system_prompt
        self.critic_prompt = critic_prompt
        self.skills = skills or list(DEFAULT_SKILLS)

    def run(
        self,
        task_graph: TaskGraph,
        capability_space: list[CapabilitySubject],
        diagnosis: dict[str, Any],
        weights: dict[str, float] | None = None,
        agent_config: AgentRuntimeConfig | dict[str, Any] | None = None,
    ) -> WorkflowPlan:
        """Deterministic five-stage fallback used by tests and offline deployments."""
        if not capability_space:
            raise ValueError("capability space cannot be empty")
        config = self._normalise_config(agent_config)
        policy = {**DEFAULT_POLICY, **config.get("parameters", {})}
        objective = {**DEFAULT_WEIGHTS, **(weights or {})}
        human_vector = diagnosis.get("planning_capability", {})
        policy["personalization_enabled"] = bool(human_vector) and diagnosis.get("method") != "assessment_disabled"
        objective = self._objective_for_mode(objective, policy["personalization_enabled"])
        subjects = self._calibrate_humans(capability_space, diagnosis)
        candidates = self._algorithmic_candidates(task_graph, subjects, human_vector, objective, policy)
        return self._complete_pipeline(
            task_graph, candidates, subjects, human_vector, diagnosis, objective, policy, config,
            generation_meta={"mode": "algorithm_fallback", "reason": "model_not_invoked"},
            critic_meta={"mode": "deterministic"},
        )

    async def run_async(
        self,
        task_graph: TaskGraph,
        capability_space: list[CapabilitySubject],
        diagnosis: dict[str, Any],
        weights: dict[str, float] | None = None,
        model_config: dict[str, Any] | None = None,
        agent_config: AgentRuntimeConfig | dict[str, Any] | None = None,
    ) -> WorkflowPlan:
        """Run the full hybrid pipeline, using the configured LLM for proposal and critique."""
        if not capability_space:
            raise ValueError("capability space cannot be empty")
        config = self._normalise_config(agent_config)
        policy = {**DEFAULT_POLICY, **config.get("parameters", {})}
        objective = {**DEFAULT_WEIGHTS, **(weights or {})}
        human_vector = diagnosis.get("planning_capability", {})
        policy["personalization_enabled"] = bool(human_vector) and diagnosis.get("method") != "assessment_disabled"
        objective = self._objective_for_mode(objective, policy["personalization_enabled"])
        subjects = self._calibrate_humans(capability_space, diagnosis)
        candidates = self._algorithmic_candidates(task_graph, subjects, human_vector, objective, policy)
        generation_meta: dict[str, Any] = {"mode": "algorithm_fallback"}
        critic_meta: dict[str, Any] = {"mode": "deterministic"}

        if model_config and model_config.get("api_key"):
            try:
                proposals = await self._generate_llm_candidates(
                    task_graph, subjects, diagnosis, model_config, config,
                )
                candidates.extend(
                    self._proposal_to_candidate(item, task_graph, subjects, human_vector, objective, policy)
                    for item in proposals.candidates
                )
                generation_meta = {
                    "mode": "llm",
                    "model": model_config.get("model"),
                    "candidate_count": len(proposals.candidates),
                }
            except Exception as exc:
                logger.warning("LLM candidate generation failed; using algorithm candidates", exc_info=True)
                generation_meta = {
                    "mode": "algorithm_fallback",
                    "reason": type(exc).__name__,
                    "detail": str(exc)[:300],
                }

        candidates = self._deduplicate_candidates(candidates)
        for candidate in candidates:
            candidate.issues = self._deterministic_critique(candidate, task_graph, subjects, human_vector, policy)

        if generation_meta.get("mode") == "llm":
            try:
                critiques = await self._criticise_with_llm(
                    candidates, task_graph, subjects, diagnosis, model_config or {}, config,
                )
                self._merge_llm_critiques(candidates, critiques)
                critic_meta = {
                    "mode": "llm_plus_constraints",
                    "model": (model_config or {}).get("model"),
                    "review_count": len(critiques.critiques),
                }
            except Exception as exc:
                logger.warning("LLM workflow critique failed; using constraint critic", exc_info=True)
                critic_meta = {
                    "mode": "deterministic_fallback",
                    "reason": type(exc).__name__,
                    "detail": str(exc)[:300],
                }

        return self._complete_pipeline(
            task_graph, candidates, subjects, human_vector, diagnosis, objective, policy, config,
            generation_meta=generation_meta, critic_meta=critic_meta,
        )

    async def _generate_llm_candidates(
        self,
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        diagnosis: dict[str, Any],
        model_config: dict[str, Any],
        config: dict[str, Any],
    ) -> CandidateProposalSet:
        skills = self.skills + self._skills_from_config(config)
        prompt = config.get("system_prompt") or self.system_prompt
        count = max(2, min(6, int(config.get("parameters", {}).get("candidate_count", 4))))
        llm = ChatOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config.get("base_url") or None,
            model=model_config.get("model") or "gpt-4.1-mini",
            temperature=model_config.get("temperature", .2),
            timeout=90,
        ).with_structured_output(CandidateProposalSet)
        context = self._planning_context(task_graph, subjects, diagnosis)
        return await llm.ainvoke(
            f"{prompt}\n\n可用 Skills：\n{self._render_skills(skills)}\n\n"
            f"请生成 {count} 个候选方案。规划上下文：\n{context}"
        )

    async def _criticise_with_llm(
        self,
        candidates: list[PlanningCandidate],
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        diagnosis: dict[str, Any],
        model_config: dict[str, Any],
        config: dict[str, Any],
    ) -> CandidateCritiqueSet:
        skills = self.skills + self._skills_from_config(config)
        llm = ChatOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config.get("base_url") or None,
            model=model_config.get("model") or "gpt-4.1-mini",
            temperature=0,
            timeout=90,
        ).with_structured_output(CandidateCritiqueSet)
        snapshots = [self._candidate_snapshot(item) for item in candidates]
        return await llm.ainvoke(
            f"{self.critic_prompt}\n\n规划 Skills：\n{self._render_skills(skills)}\n\n"
            f"任务与能力上下文：\n{self._planning_context(task_graph, subjects, diagnosis)}\n\n"
            f"待审查候选：\n{json.dumps(snapshots, ensure_ascii=False)}"
        )

    def _complete_pipeline(
        self,
        task_graph: TaskGraph,
        candidates: list[PlanningCandidate],
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        diagnosis: dict[str, Any],
        objective: dict[str, float],
        policy: dict[str, Any],
        config: dict[str, Any],
        generation_meta: dict[str, Any],
        critic_meta: dict[str, Any],
    ) -> WorkflowPlan:
        for candidate in candidates:
            self._repair_candidate(candidate, task_graph, subjects, human_vector, objective, policy)
            remaining = self._deterministic_critique(candidate, task_graph, subjects, human_vector, policy)
            remaining_keys = {(item.get("code"), item.get("subtask_id")) for item in remaining}
            for issue in candidate.issues:
                key = (issue.get("code"), issue.get("subtask_id"))
                issue["resolved"] = issue.get("source") == "constraint_critic" and key not in remaining_keys
            candidate.issues.extend(
                issue for issue in remaining
                if (issue.get("code"), issue.get("subtask_id"))
                not in {(item.get("code"), item.get("subtask_id")) for item in candidate.issues}
            )
            self._score_plan_candidate(candidate, task_graph)
        candidates.sort(key=lambda item: item.utility, reverse=True)
        selected = candidates[0]
        plan = self._synthesise(
            task_graph, selected.assignments, subjects, human_vector, diagnosis, policy,
            selected.flow_edges, selected.human_review_subtasks,
        )
        skills = self.skills + self._skills_from_config(config)
        plan.decision_trace.insert(0, {
            "agent": self.name,
            "algorithm": self.algorithm_version,
            "prompt_version": self.prompt_version,
            "objective_weights": objective,
            "policy": policy,
            "skills": [skill.name for skill in skills if skill.enabled],
            "human_assessment": {
                "method": diagnosis.get("method"),
                "personalization_enabled": policy["personalization_enabled"],
                "overall": diagnosis.get("overall", 0),
                "confidence": diagnosis.get("confidence", 0),
                "weakest_dimensions": diagnosis.get("weakest_dimensions", []),
            },
            "pipeline": [
                {"stage": "candidate_generation", **generation_meta},
                {"stage": "multi_objective_optimisation", "candidate_count": len(candidates)},
                {"stage": "critic", **critic_meta},
                {"stage": "constraint_repair", "repair_count": sum(len(item.repairs) for item in candidates)},
                {"stage": "final_selection", "selected": selected.name, "utility": round(selected.utility, 4)},
            ],
            "candidate_decisions": [self._candidate_snapshot(item) for item in candidates],
            "reason": "由 LLM 提案、能力约束优化、Critic 审查和自动修复共同决定最终工作流。",
        })
        return plan

    def _algorithmic_candidates(
        self,
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        objective: dict[str, float],
        policy: dict[str, Any],
    ) -> list[PlanningCandidate]:
        variants = [
            ("算法基线", "balanced", objective),
            ("舒适区优先", "comfort_first", {**objective, "comfort": max(.32, objective["comfort"]), "cost": objective["cost"] * .6}),
            ("效率优先", "efficiency_first", {**objective, "cost": max(.18, objective["cost"]), "latency": max(.14, objective["latency"])}),
        ]
        return [
            PlanningCandidate(
                name=name,
                strategy=strategy,
                assignments=self._optimise(task_graph, subjects, human_vector, variant, policy),
                rationale="由可复现的全局 Beam Search 生成，作为 LLM 候选的约束基线。",
            )
            for name, strategy, variant in variants
        ]

    def _proposal_to_candidate(
        self,
        proposal: CandidateProposal,
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        objective: dict[str, float],
        policy: dict[str, Any],
    ) -> PlanningCandidate:
        task_by_id = {item.id: item for item in task_graph.subtasks}
        subject_by_id = {item.id: item for item in subjects}
        assignments: list[AssignmentCandidate] = []
        used: set[str] = set()
        for item in proposal.assignments:
            if item.subtask_id in used or item.subtask_id not in task_by_id or item.subject_id not in subject_by_id:
                continue
            assignments.append(self._score_candidate(
                task_by_id[item.subtask_id], subject_by_id[item.subject_id], human_vector, objective, policy,
            ))
            used.add(item.subtask_id)
        return PlanningCandidate(
            name=proposal.name,
            strategy=proposal.strategy,
            assignments=assignments,
            flow_edges=list(proposal.edges),
            human_review_subtasks=set(proposal.human_review_subtask_ids),
            rationale=proposal.rationale,
            source="llm",
        )

    @staticmethod
    def _deduplicate_candidates(candidates: list[PlanningCandidate]) -> list[PlanningCandidate]:
        seen: set[tuple[tuple[str, str], ...]] = set()
        result: list[PlanningCandidate] = []
        for candidate in candidates:
            signature = tuple(sorted((item.subtask.id, item.subject.id) for item in candidate.assignments))
            if signature in seen and candidate.source == "algorithm":
                continue
            seen.add(signature)
            result.append(candidate)
        return result

    def _deterministic_critique(
        self,
        candidate: PlanningCandidate,
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        task_ids = {item.id for item in task_graph.subtasks}
        subject_ids = {item.id for item in subjects}
        assignments = {item.subtask.id: item for item in candidate.assignments}
        for task in task_graph.subtasks:
            selected = assignments.get(task.id)
            if not selected:
                issues.append(self._issue("missing_assignment", "error", task.id, "子任务没有分配主体", "分配最高可行得分主体"))
                continue
            if selected.subject.id not in subject_ids:
                issues.append(self._issue("invalid_subject", "error", task.id, "使用了不可用主体", "替换为可用主体"))
            if selected.subject.subject_type in task.unsuitable_subject_types:
                issues.append(self._issue("unsuitable_subject", "error", task.id, "主体类型被问题分析标记为不适合", "选择下一可行主体"))
            if selected.subject.subject_type == SubjectType.HUMAN and selected.overload > float(policy["human_overload_limit"]):
                issues.append(self._issue("human_overload", "warning", task.id, "任务超出用户能力舒适区", "由机器接管或提供结构化支架"))
            if (
                task.risk >= float(policy["human_review_risk"])
                and selected.subject.subject_type != SubjectType.HUMAN
                and task.id not in candidate.human_review_subtasks
            ):
                issues.append(self._issue("missing_human_review", "error", task.id, "高风险机器节点缺少人类确认", "增加人类责任确认节点"))

        for edge in candidate.flow_edges:
            if edge.source_subtask_id not in task_ids or edge.target_subtask_id not in task_ids:
                issues.append(self._issue("invalid_edge", "error", edge.target_subtask_id, "边引用了不存在的子任务", "删除无效边"))
            if edge.edge_type in {EdgeType.CONDITIONAL, EdgeType.LOOP} and not edge.condition:
                issues.append(self._issue("missing_condition", "error", edge.target_subtask_id, "条件边或循环没有触发条件", "补充可执行条件"))
            if edge.edge_type == EdgeType.LOOP and edge.max_iterations < 1:
                issues.append(self._issue("unbounded_loop", "error", edge.target_subtask_id, "反馈循环没有有效上限", "设置最大迭代次数"))
        return issues

    @staticmethod
    def _issue(code: str, severity: str, subtask_id: str | None, description: str, action: str) -> dict[str, Any]:
        return {
            "code": code,
            "severity": severity,
            "subtask_id": subtask_id,
            "description": description,
            "suggested_action": action,
            "source": "constraint_critic",
        }

    @staticmethod
    def _merge_llm_critiques(candidates: list[PlanningCandidate], critiques: CandidateCritiqueSet) -> None:
        by_name = {item.name: item for item in candidates}
        for review in critiques.critiques:
            candidate = by_name.get(review.candidate_name)
            if not candidate:
                continue
            existing = {(item.get("code"), item.get("subtask_id")) for item in candidate.issues}
            for issue in review.issues:
                key = (issue.code, issue.subtask_id)
                if key in existing:
                    continue
                candidate.issues.append({**issue.model_dump(), "source": "llm_critic"})

    def _repair_candidate(
        self,
        candidate: PlanningCandidate,
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        objective: dict[str, float],
        policy: dict[str, Any],
    ) -> None:
        task_by_id = {item.id: item for item in task_graph.subtasks}
        assignments = {item.subtask.id: item for item in candidate.assignments if item.subtask.id in task_by_id}
        ranked = {
            task.id: sorted(
                [self._score_candidate(task, subject, human_vector, objective, policy) for subject in subjects],
                key=lambda item: item.score,
                reverse=True,
            )
            for task in task_graph.subtasks
        }
        for task in task_graph.subtasks:
            selected = assignments.get(task.id)
            if selected is None:
                assignments[task.id] = ranked[task.id][0]
                candidate.repairs.append({"code": "fill_assignment", "subtask_id": task.id, "selected": ranked[task.id][0].subject.id})
                continue
            if selected.subject.subject_type in task.unsuitable_subject_types:
                replacement = next(
                    (item for item in ranked[task.id] if item.subject.subject_type not in task.unsuitable_subject_types),
                    ranked[task.id][0],
                )
                assignments[task.id] = replacement
                candidate.repairs.append({"code": "replace_unsuitable_subject", "subtask_id": task.id, "selected": replacement.subject.id})
                selected = replacement
            if (
                selected.subject.subject_type == SubjectType.HUMAN
                and selected.overload > float(policy["human_overload_limit"])
                and task.risk < float(policy["human_review_risk"])
            ):
                machine = next(
                    (item for item in ranked[task.id] if item.subject.subject_type != SubjectType.HUMAN and item.score >= selected.score - .08),
                    None,
                )
                if machine:
                    assignments[task.id] = machine
                    candidate.repairs.append({"code": "protect_human_comfort", "subtask_id": task.id, "selected": machine.subject.id})

        candidate.assignments = [assignments[item.id] for item in task_graph.subtasks]
        valid_ids = set(task_by_id)
        repaired_edges: list[ProposedFlowEdge] = []
        seen_edges: set[tuple[str, str, EdgeType]] = set()
        for edge in candidate.flow_edges:
            if edge.source_subtask_id not in valid_ids or edge.target_subtask_id not in valid_ids:
                candidate.repairs.append({"code": "remove_invalid_edge", "source": edge.source_subtask_id, "target": edge.target_subtask_id})
                continue
            key = (edge.source_subtask_id, edge.target_subtask_id, edge.edge_type)
            if key in seen_edges:
                continue
            condition = edge.condition
            if edge.edge_type in {EdgeType.CONDITIONAL, EdgeType.LOOP} and not condition:
                condition = "needs_revision == true" if edge.edge_type == EdgeType.LOOP else "confidence < 0.7"
                candidate.repairs.append({"code": "add_edge_condition", "target": edge.target_subtask_id, "condition": condition})
            repaired_edges.append(edge.model_copy(update={"condition": condition, "max_iterations": max(1, min(10, edge.max_iterations))}))
            seen_edges.add(key)
        candidate.flow_edges = repaired_edges

        has_human = any(item.subject_type == SubjectType.HUMAN for item in subjects)
        if has_human:
            for task in task_graph.subtasks:
                selected = assignments[task.id]
                if (
                    task.risk >= float(policy["human_review_risk"])
                    and selected.subject.subject_type != SubjectType.HUMAN
                    and task.id not in candidate.human_review_subtasks
                ):
                    candidate.human_review_subtasks.add(task.id)
                    candidate.repairs.append({"code": "add_human_review", "subtask_id": task.id})
        candidate.human_review_subtasks.intersection_update(valid_ids)

    @staticmethod
    def _score_plan_candidate(candidate: PlanningCandidate, task_graph: TaskGraph) -> None:
        count = max(1, len(task_graph.subtasks))
        assignment_quality = sum(item.score for item in candidate.assignments) / count
        subject_types = {item.subject.subject_type for item in candidate.assignments}
        diversity = min(1.0, len(subject_types) / max(1, min(4, count)))
        desired_non_linear = sum(item.execution_mode != ExecutionMode.SEQUENTIAL for item in task_graph.subtasks)
        proposed_non_linear = sum(item.edge_type != EdgeType.DEFAULT for item in candidate.flow_edges)
        topology = 1.0 if desired_non_linear == 0 else min(1.0, proposed_non_linear / desired_non_linear)
        unresolved = [item for item in candidate.issues if not item.get("resolved", False)]
        error_count = sum(item.get("severity") == "error" for item in unresolved)
        warning_count = sum(item.get("severity") != "error" for item in unresolved)
        repair_penalty = min(.12, len(candidate.repairs) * .012)
        issue_penalty = min(.3, error_count * .06 + warning_count * .018)
        candidate.metrics = {
            "assignment_quality": round(assignment_quality, 4),
            "subject_diversity": round(diversity, 4),
            "topology_coverage": round(topology, 4),
            "issue_penalty": round(issue_penalty, 4),
            "repair_penalty": round(repair_penalty, 4),
        }
        candidate.utility = max(0.0, min(1.0, assignment_quality * .78 + diversity * .1 + topology * .12 - issue_penalty - repair_penalty))

    @staticmethod
    def _candidate_snapshot(candidate: PlanningCandidate) -> dict[str, Any]:
        return {
            "name": candidate.name,
            "source": candidate.source,
            "strategy": candidate.strategy,
            "rationale": candidate.rationale,
            "assignments": [
                {
                    "subtask_id": item.subtask.id,
                    "subject_id": item.subject.id,
                    "subject_type": item.subject.subject_type.value,
                    "score": round(item.score, 4),
                    "evidence": item.explanation,
                }
                for item in candidate.assignments
            ],
            "flow_edges": [item.model_dump(mode="json") for item in candidate.flow_edges],
            "human_review_subtasks": sorted(candidate.human_review_subtasks),
            "critic_issues": candidate.issues,
            "repairs": candidate.repairs,
            "metrics": candidate.metrics,
            "utility": round(candidate.utility, 4),
        }

    @staticmethod
    def _planning_context(
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        diagnosis: dict[str, Any],
    ) -> str:
        safe_diagnosis = {
            "method": diagnosis.get("method"),
            "overall": diagnosis.get("overall"),
            "confidence": diagnosis.get("confidence"),
            "weakest_dimensions": diagnosis.get("weakest_dimensions", []),
            "planning_capability": diagnosis.get("planning_capability", {}),
        }
        safe_subjects = [
            {
                "id": item.id,
                "name": item.name,
                "subject_type": item.subject_type.value,
                "capability": item.capability,
                "reliability": item.reliability,
                "cost": item.cost,
                "latency": item.latency,
            }
            for item in subjects
        ]
        return json.dumps(
            {
                "task_graph": task_graph.model_dump(mode="json"),
                "cognitive_diagnosis": safe_diagnosis,
                "available_subjects": safe_subjects,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _skills_from_config(config: dict[str, Any]) -> list[AgentSkill]:
        return [
            item if isinstance(item, AgentSkill) else AgentSkill.model_validate(item)
            for item in config.get("skills", [])
        ]

    @staticmethod
    def _render_skills(skills: list[AgentSkill]) -> str:
        return "\n".join(
            f"- {item.name}: {item.description}。{item.instructions}"
            for item in skills
            if item.enabled
        )

    @staticmethod
    def _calibrate_humans(subjects: list[CapabilitySubject], diagnosis: dict[str, Any]) -> list[CapabilitySubject]:
        human_vector = diagnosis.get("planning_capability", {})
        calibrated = []
        for subject in subjects:
            if subject.subject_type != SubjectType.HUMAN:
                calibrated.append(subject)
                continue
            calibrated.append(subject.model_copy(update={
                "capability": {**subject.capability, **human_vector},
                "metadata": {
                    **subject.metadata,
                    "diagnostic_method": diagnosis.get("method"),
                    "assessment_confidence": diagnosis.get("confidence", 0),
                    "assessment_overall": diagnosis.get("overall", 0),
                },
            }))
        return calibrated

    def _optimise(
        self,
        task_graph: TaskGraph,
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        weights: dict[str, float],
        policy: dict[str, Any],
    ) -> list[AssignmentCandidate]:
        candidate_sets = [
            sorted(
                [self._score_candidate(item, subject, human_vector, weights, policy) for subject in subjects],
                key=lambda candidate: candidate.score,
                reverse=True,
            )[: int(policy["max_candidates_per_task"])]
            for item in task_graph.subtasks
        ]
        beam = [BeamState()]
        for candidates in candidate_sets:
            expanded: list[BeamState] = []
            for state in beam:
                for candidate in candidates:
                    subject_load = state.loads.get(candidate.subject.id, 0)
                    type_key = candidate.subject.subject_type.value
                    diversity = float(policy["type_diversity_bonus"]) if not state.type_loads.get(type_key) else 0
                    load_penalty = float(policy["load_balance_penalty"]) * subject_load
                    utility = state.utility + candidate.score + diversity - load_penalty
                    expanded.append(BeamState(
                        assignments=[*state.assignments, candidate],
                        loads={**state.loads, candidate.subject.id: subject_load + 1},
                        type_loads={**state.type_loads, type_key: state.type_loads.get(type_key, 0) + 1},
                        utility=utility,
                    ))
            beam = sorted(expanded, key=lambda state: state.utility, reverse=True)[: int(policy["beam_width"])]
        if not beam:
            raise ValueError("no feasible subject assignment")
        return beam[0].assignments

    def _score_candidate(
        self,
        subtask: Subtask,
        subject: CapabilitySubject,
        human_vector: dict[str, float],
        weights: dict[str, float],
        policy: dict[str, Any],
    ) -> AssignmentCandidate:
        requirement = subtask.requirement.model_dump()
        active = {name: value for name, value in requirement.items() if value > 0}
        mass = sum(active.values()) or 1
        fit = sum(weight * min(1.0, subject.capability.get(name, 0) / max(weight, .01)) for name, weight in active.items()) / mass
        personalized = bool(policy.get("personalization_enabled", True))
        human_gap = (
            sum(weight * (weight - human_vector.get(name, .5)) for name, weight in active.items()) / mass
            if personalized else 0.0
        )
        overload = (
            max((need - human_vector.get(name, .5) for name, need in active.items()), default=0)
            if personalized else 0.0
        )
        target = float(policy["comfort_target_gap"])
        bandwidth = max(.05, float(policy["comfort_bandwidth"]))
        human_comfort = exp(-((human_gap - target) / bandwidth) ** 2)
        complementarity = (
            sum(
                need * max(0.0, min(need - human_vector.get(name, .5), subject.capability.get(name, 0) - human_vector.get(name, .5)))
                for name, need in active.items()
            ) / mass
            if personalized else 0.0
        )
        suitability = {fit.subject_type: fit.suitability for fit in subtask.subject_suitability}
        type_prior = suitability.get(subject.subject_type, self._default_type_prior(subtask.task_type, subject.subject_type))
        comfort = (
            human_comfort if subject.subject_type == SubjectType.HUMAN else min(1.0, .72 + .28 * max(0, complementarity))
        ) if personalized else 1.0
        score = (
            weights["fit"] * (.72 * fit + .28 * type_prior)
            + weights["reliability"] * subject.reliability
            + weights["comfort"] * comfort
            + weights["complementarity"] * max(0, complementarity)
            - weights["cost"] * subject.cost
            - weights["latency"] * subject.latency
        )
        if subject.subject_type in subtask.preferred_subject_types:
            score += .07
        if subject.subject_type in subtask.unsuitable_subject_types:
            score -= .22
        if subtask.risk >= .75 and subject.subject_type == SubjectType.HUMAN:
            score += .1
        if personalized and subject.subject_type == SubjectType.HUMAN and overload > float(policy["human_overload_limit"]):
            score -= .3 * min(1.0, overload)
        score = max(0.0, min(1.0, score))
        explanation = (
            f"fit={fit:.3f}, subject_prior={type_prior:.3f}, comfort={comfort:.3f}, "
            f"human_gap={human_gap:.3f}, complementarity={complementarity:.3f}, overload={overload:.3f}"
        )
        return AssignmentCandidate(subtask, subject, score, fit, type_prior, comfort, human_gap, complementarity, overload, explanation)

    def _synthesise(
        self,
        task_graph: TaskGraph,
        assignments: list[AssignmentCandidate],
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
        diagnosis: dict[str, Any],
        policy: dict[str, Any],
        proposed_edges: list[ProposedFlowEdge] | None = None,
        requested_human_reviews: set[str] | None = None,
    ) -> WorkflowPlan:
        nodes: list[WorkflowNode] = []
        edges: list[WorkflowEdge] = []
        trace: list[dict[str, Any]] = []
        main_node: dict[str, str] = {}
        exit_node: dict[str, str] = {}
        total_cost = 0.0
        human_subjects = [subject for subject in subjects if subject.subject_type == SubjectType.HUMAN]
        requested_human_reviews = requested_human_reviews or set()

        for candidate in assignments:
            item = candidate.subtask
            node_id = f"node-{item.id}"
            main_node[item.id] = node_id
            exit_node[item.id] = node_id
            support = self._best_machine_support(item, subjects, human_vector)
            config = self._node_config(candidate, support, diagnosis)
            nodes.append(WorkflowNode(
                id=node_id,
                subtask_id=item.id,
                subject_id=candidate.subject.id,
                subject_type=candidate.subject.subject_type,
                label=item.name,
                match_score=round(candidate.score, 4),
                config=config,
                explainability={
                    "reason": candidate.explanation,
                    "problem_analysis_rationale": item.assignment_rationale,
                    "fit": round(candidate.fit, 4),
                    "subject_type_prior": round(candidate.type_prior, 4),
                    "human_comfort_score": round(candidate.comfort, 4),
                    "human_capability_gap": round(candidate.human_gap, 4),
                    "machine_complementarity": round(candidate.complementarity, 4),
                    "human_overload": round(candidate.overload, 4),
                },
            ))
            total_cost += candidate.subject.cost
            trace.append({
                "subtask_id": item.id,
                "selected": candidate.subject.id,
                "subject_type": candidate.subject.subject_type.value,
                "score": round(candidate.score, 4),
                "metrics": {
                    "fit": round(candidate.fit, 4),
                    "comfort": round(candidate.comfort, 4),
                    "human_gap": round(candidate.human_gap, 4),
                    "complementarity": round(candidate.complementarity, 4),
                    "overload": round(candidate.overload, 4),
                },
                "control_flow": item.execution_mode.value,
                "reason": candidate.explanation,
            })

            needs_review = (
                item.id in requested_human_reviews
                or item.risk >= float(policy["human_review_risk"])
            )
            if needs_review and candidate.subject.subject_type != SubjectType.HUMAN and human_subjects:
                reviewer = human_subjects[0]
                review_id = f"{node_id}-human-check"
                nodes.append(WorkflowNode(
                    id=review_id,
                    subtask_id=f"{item.id}-human-check",
                    subject_id=reviewer.id,
                    subject_type=SubjectType.HUMAN,
                    label=f"{item.name} · 人类责任确认",
                    match_score=round(max(.45, candidate.comfort), 4),
                    config={
                        "instruction": f"依据上游证据复核“{item.description}”，只判断高风险要点并记录理由。",
                        "approval_criteria": "风险可接受、证据充分、责任明确",
                        "comfort_scaffold": support,
                        "diagnostic_confidence": diagnosis.get("confidence", 0),
                    },
                    explainability={
                        "reason": "主执行由机器完成以覆盖能力差距；高风险最终责任保留给人类，并提供结构化支架。",
                        "human_comfort_zone": True,
                    },
                ))
                edges.append(WorkflowEdge(source=node_id, target=review_id))
                exit_node[item.id] = review_id
                total_cost += reviewer.cost

        def append_edge(edge: WorkflowEdge) -> None:
            for index, existing in enumerate(edges):
                if existing.source == edge.source and existing.target == edge.target:
                    if existing.edge_type == EdgeType.DEFAULT and edge.edge_type != EdgeType.DEFAULT:
                        edges[index] = edge
                    return
            edges.append(edge)

        for candidate in assignments:
            item = candidate.subtask
            for dependency in item.dependencies:
                if dependency not in exit_node:
                    continue
                edge_type = EdgeType.CONDITIONAL if item.entry_condition else EdgeType.DEFAULT
                append_edge(WorkflowEdge(
                    source=exit_node[dependency],
                    target=main_node[item.id],
                    condition=item.entry_condition,
                    edge_type=edge_type,
                ))
            policy_item = item.iteration_policy
            target = policy_item.feedback_target_subtask_id
            if policy_item.enabled and target in main_node:
                append_edge(WorkflowEdge(
                    source=exit_node[item.id],
                    target=main_node[target],
                    condition=policy_item.condition,
                    edge_type=EdgeType.LOOP,
                    max_iterations=policy_item.max_iterations,
                ))

        for proposed in proposed_edges or []:
            if proposed.source_subtask_id not in exit_node or proposed.target_subtask_id not in main_node:
                continue
            append_edge(WorkflowEdge(
                source=exit_node[proposed.source_subtask_id],
                target=main_node[proposed.target_subtask_id],
                condition=proposed.condition,
                edge_type=proposed.edge_type,
                max_iterations=proposed.max_iterations,
            ))

        flow_summary = {
            "parallel_fan_outs": self._count_fan_outs(edges),
            "conditional_edges": sum(edge.edge_type == EdgeType.CONDITIONAL for edge in edges),
            "loop_edges": sum(edge.edge_type == EdgeType.LOOP for edge in edges),
        }
        trace.insert(0, {"workflow_topology": flow_summary})
        return WorkflowPlan(
            id=str(uuid4()),
            task_id=task_graph.task_id,
            nodes=nodes,
            edges=edges,
            decision_trace=trace,
            estimated_cost=round(total_cost, 4),
            requires_human=any(node.subject_type == SubjectType.HUMAN for node in nodes),
        )

    @staticmethod
    def _node_config(candidate: AssignmentCandidate, support: dict[str, Any] | None, diagnosis: dict[str, Any]) -> dict[str, Any]:
        item = candidate.subtask
        base: dict[str, Any] = {"timeout_seconds": 60, "retry": 1, "execution_mode": item.execution_mode.value}
        if candidate.subject.subject_type == SubjectType.LLM:
            base.update({
                "system_prompt": "你是可审计的任务执行智能体。引用上游证据，标注不确定性，并严格遵守输出契约。",
                "prompt_template": f"任务：{item.description}\n用户输入：{{input}}\n上游结果：{{upstream}}",
                "temperature": .2,
            })
        elif candidate.subject.subject_type == SubjectType.ML:
            base.update({
                "runtime": "python",
                "requirements": ["numpy", "scikit-learn"],
                "code": "def run(payload, upstream):\n    # 加载已验证模型并返回预测、置信度和版本\n    return {'prediction': None, 'confidence': None, 'upstream': upstream}\n",
            })
        elif candidate.subject.subject_type == SubjectType.HUMAN:
            base.update({
                "instruction": f"请完成“{item.description}”，依据检查表记录证据、判断和不确定性。",
                "approval_criteria": "准确、可解释、符合领域规范并明确责任",
                "comfort_scaffold": support,
                "diagnosed_weaknesses": diagnosis.get("weakest_dimensions", []),
            })
        else:
            base.update({"connector": "passthrough", "operation": item.task_type})
        return base

    def _best_machine_support(
        self,
        item: Subtask,
        subjects: list[CapabilitySubject],
        human_vector: dict[str, float],
    ) -> dict[str, Any] | None:
        if not human_vector:
            return None
        machines = [subject for subject in subjects if subject.subject_type != SubjectType.HUMAN]
        if not machines:
            return None
        requirement = item.requirement.model_dump()
        gaps = {name: max(0, need - human_vector.get(name, .5)) for name, need in requirement.items() if need > 0}
        scored = [
            (sum(gap * subject.capability.get(name, 0) for name, gap in gaps.items()), subject)
            for subject in machines
        ]
        score, subject = max(scored, key=lambda pair: pair[0])
        if score <= 0:
            return None
        return {
            "subject_id": subject.id,
            "subject_type": subject.subject_type.value,
            "purpose": "为用户薄弱能力提供证据、草稿或计算支持，最终判断仍由用户完成",
            "covered_gaps": [name for name, gap in gaps.items() if gap > 0 and subject.capability.get(name, 0) >= .6],
        }

    @staticmethod
    def _default_type_prior(task_type: str, subject_type: SubjectType) -> float:
        preferred = {
            "data_processing": SubjectType.TOOL,
            "prediction": SubjectType.ML,
            "generation": SubjectType.LLM,
            "reasoning": SubjectType.LLM,
            "human_review": SubjectType.HUMAN,
            "tool": SubjectType.TOOL,
        }.get(task_type, SubjectType.LLM)
        return .85 if subject_type == preferred else .45

    @staticmethod
    def _count_fan_outs(edges: list[WorkflowEdge]) -> int:
        counts: dict[str, int] = {}
        for edge in edges:
            if edge.edge_type != EdgeType.LOOP:
                counts[edge.source] = counts.get(edge.source, 0) + 1
        return sum(count > 1 for count in counts.values())

    @staticmethod
    def _objective_for_mode(weights: dict[str, float], personalized: bool) -> dict[str, float]:
        if personalized:
            return weights
        result = dict(weights)
        personalization_mass = result.get("comfort", 0) + result.get("complementarity", 0)
        result["comfort"] = 0.0
        result["complementarity"] = 0.0
        result["fit"] = result.get("fit", 0) + personalization_mass * .7
        result["reliability"] = result.get("reliability", 0) + personalization_mass * .3
        return result

    @staticmethod
    def _normalise_config(config: AgentRuntimeConfig | dict[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        raw = config.model_dump() if isinstance(config, AgentRuntimeConfig) else dict(config)
        raw["skills"] = [skill.model_dump() if hasattr(skill, "model_dump") else skill for skill in raw.get("skills", [])]
        return raw
