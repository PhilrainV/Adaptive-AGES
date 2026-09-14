from uuid import uuid4
from app.schemas.domain import PlanRequest, WorkflowEdge, WorkflowNode, WorkflowPlan
from app.services.capability_matching import CapabilityMatcher


class AdaptivePlanner:
    def __init__(self, matcher: CapabilityMatcher | None = None):
        self.matcher = matcher or CapabilityMatcher()

    def plan(self, request: PlanRequest) -> WorkflowPlan:
        nodes: list[WorkflowNode] = []
        edges: list[WorkflowEdge] = []
        trace: list[dict] = []
        subtask_to_node: dict[str, str] = {}
        total_cost = 0.0

        for subtask in request.task_graph.subtasks:
            candidates = self.matcher.rank(subtask, request.capability_space, request.weights, request.user_profile)
            if not candidates:
                raise ValueError(f"No capable subject is available for subtask {subtask.id}")
            chosen = candidates[0]
            node_id = f"node-{subtask.id}"
            subtask_to_node[subtask.id] = node_id
            subject = next(s for s in request.capability_space if s.id == chosen.subject_id)
            total_cost += subject.cost
            nodes.append(WorkflowNode(id=node_id, subtask_id=subtask.id, subject_id=chosen.subject_id, subject_type=chosen.subject_type, label=subtask.name, match_score=chosen.score, config={"timeout_seconds": 60, "retry": 1}, explainability={"reason": chosen.explanation, "similarity": chosen.similarity, "alternatives": chosen.alternatives}))
            trace.append({"subtask_id": subtask.id, "selected": chosen.subject_id, "subject_type": chosen.subject_type, "score": chosen.score, "reason": chosen.explanation, "alternatives": chosen.alternatives})

        for subtask in request.task_graph.subtasks:
            for dependency in subtask.dependencies:
                if dependency in subtask_to_node:
                    edges.append(WorkflowEdge(source=subtask_to_node[dependency], target=subtask_to_node[subtask.id]))

        return WorkflowPlan(id=str(uuid4()), task_id=request.task_graph.task_id, nodes=nodes, edges=edges, decision_trace=trace, estimated_cost=round(total_cost, 4), requires_human=any(n.subject_type.value == "human" for n in nodes))
