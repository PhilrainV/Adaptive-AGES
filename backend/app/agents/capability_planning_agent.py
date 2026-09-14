from app.schemas.domain import CapabilitySubject, PlanRequest, SubjectType, TaskGraph, WorkflowPlan
from app.services.adaptive_planner import AdaptivePlanner


class CapabilityPlanningAgent:
    """Agent 4: plans a workflow using the task graph and diagnosed human ability."""

    name = "capability_planning_agent"

    def __init__(self, planner: AdaptivePlanner | None = None):
        self.planner = planner or AdaptivePlanner()

    def run(
        self,
        task_graph: TaskGraph,
        capability_space: list[CapabilitySubject],
        diagnosis: dict,
        weights: dict[str, float] | None = None,
    ) -> WorkflowPlan:
        human_vector = diagnosis.get("planning_capability", {})
        calibrated_subjects = []
        for subject in capability_space:
            if subject.subject_type == SubjectType.HUMAN:
                calibrated_subjects.append(
                    subject.model_copy(
                        update={
                            "capability": {**subject.capability, **human_vector},
                            "metadata": {
                                **subject.metadata,
                                "assessment_confidence": diagnosis.get("confidence", 0),
                                "assessment_overall": diagnosis.get("overall", 0),
                            },
                        }
                    )
                )
            else:
                calibrated_subjects.append(subject)
        request = PlanRequest(
            task_graph=task_graph,
            capability_space=calibrated_subjects,
            user_profile=human_vector,
            weights=weights
            or {"fit": 0.65, "reliability": 0.2, "cost": 0.1, "latency": 0.05},
        )
        plan = self.planner.plan(request)
        plan.decision_trace.insert(
            0,
            {
                "agent": self.name,
                "human_assessment_overall": diagnosis.get("overall", 0),
                "human_assessment_confidence": diagnosis.get("confidence", 0),
                "reason": "规划前已用本次任务测试结果校准 Human Agent 能力。",
            },
        )
        return plan
