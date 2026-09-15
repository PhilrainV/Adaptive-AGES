"""Agent 4: comfort-aware global allocation and non-linear workflow synthesis.

The optimisation policy, objective weights and node templates live in this file so
researchers can replace individual components without touching API routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any
from uuid import uuid4

from app.schemas.domain import (
    AgentRuntimeConfig,
    CapabilitySubject,
    EdgeType,
    SubjectType,
    Subtask,
    TaskGraph,
    WorkflowEdge,
    WorkflowNode,
    WorkflowPlan,
)

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

DEFAULT_SKILLS = [
    "global_assignment_optimisation",
    "human_comfort_zone_protection",
    "machine_complementarity",
    "risk_governance",
    "conditional_and_iterative_flow_synthesis",
]


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


class CapabilityPlanningAgent:
    """Balance machine capability and diagnosed human mastery across the full graph."""

    name = "capability_planning_agent"
    algorithm_version = "comfort-beam-2.0"

    def run(
        self,
        task_graph: TaskGraph,
        capability_space: list[CapabilitySubject],
        diagnosis: dict[str, Any],
        weights: dict[str, float] | None = None,
        agent_config: AgentRuntimeConfig | dict[str, Any] | None = None,
    ) -> WorkflowPlan:
        if not capability_space:
            raise ValueError("capability space cannot be empty")
        config = self._normalise_config(agent_config)
        policy = {**DEFAULT_POLICY, **config.get("parameters", {})}
        objective = {**DEFAULT_WEIGHTS, **(weights or {})}
        human_vector = diagnosis.get("planning_capability", {})
        subjects = self._calibrate_humans(capability_space, diagnosis)
        assignments = self._optimise(task_graph, subjects, human_vector, objective, policy)
        plan = self._synthesise(task_graph, assignments, subjects, human_vector, diagnosis, policy)
        plan.decision_trace.insert(0, {
            "agent": self.name,
            "algorithm": self.algorithm_version,
            "objective_weights": objective,
            "policy": policy,
            "skills": DEFAULT_SKILLS + [skill["name"] for skill in config.get("skills", []) if skill.get("enabled", True)],
            "human_assessment": {
                "method": diagnosis.get("method"),
                "overall": diagnosis.get("overall", 0),
                "confidence": diagnosis.get("confidence", 0),
                "weakest_dimensions": diagnosis.get("weakest_dimensions", []),
            },
            "reason": "使用全局优化平衡任务适配、用户舒适区、机器补偿、风险、成本、时延与主体负载。",
        })
        return plan

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
        human_gap = sum(weight * (weight - human_vector.get(name, .5)) for name, weight in active.items()) / mass
        overload = max((need - human_vector.get(name, .5) for name, need in active.items()), default=0)
        target = float(policy["comfort_target_gap"])
        bandwidth = max(.05, float(policy["comfort_bandwidth"]))
        human_comfort = exp(-((human_gap - target) / bandwidth) ** 2)
        complementarity = sum(
            need * max(0.0, min(need - human_vector.get(name, .5), subject.capability.get(name, 0) - human_vector.get(name, .5)))
            for name, need in active.items()
        ) / mass
        suitability = {fit.subject_type: fit.suitability for fit in subtask.subject_suitability}
        type_prior = suitability.get(subject.subject_type, self._default_type_prior(subtask.task_type, subject.subject_type))
        comfort = human_comfort if subject.subject_type == SubjectType.HUMAN else min(1.0, .72 + .28 * max(0, complementarity))
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
        if subject.subject_type == SubjectType.HUMAN and overload > float(policy["human_overload_limit"]):
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
    ) -> WorkflowPlan:
        nodes: list[WorkflowNode] = []
        edges: list[WorkflowEdge] = []
        trace: list[dict[str, Any]] = []
        main_node: dict[str, str] = {}
        exit_node: dict[str, str] = {}
        total_cost = 0.0
        human_subjects = [subject for subject in subjects if subject.subject_type == SubjectType.HUMAN]

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

            if item.risk >= float(policy["human_review_risk"]) and candidate.subject.subject_type != SubjectType.HUMAN and human_subjects:
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

        for candidate in assignments:
            item = candidate.subtask
            for dependency in item.dependencies:
                if dependency not in exit_node:
                    continue
                edge_type = EdgeType.CONDITIONAL if item.entry_condition else EdgeType.DEFAULT
                edges.append(WorkflowEdge(
                    source=exit_node[dependency],
                    target=main_node[item.id],
                    condition=item.entry_condition,
                    edge_type=edge_type,
                ))
            policy_item = item.iteration_policy
            target = policy_item.feedback_target_subtask_id
            if policy_item.enabled and target in main_node:
                edges.append(WorkflowEdge(
                    source=exit_node[item.id],
                    target=main_node[target],
                    condition=policy_item.condition,
                    edge_type=EdgeType.LOOP,
                    max_iterations=policy_item.max_iterations,
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
    def _normalise_config(config: AgentRuntimeConfig | dict[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        raw = config.model_dump() if isinstance(config, AgentRuntimeConfig) else dict(config)
        raw["skills"] = [skill.model_dump() if hasattr(skill, "model_dump") else skill for skill in raw.get("skills", [])]
        return raw
