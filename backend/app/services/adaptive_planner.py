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
            config = {"timeout_seconds": 60, "retry": 1}
            if chosen.subject_type.value == "llm":
                config.update({
                    "system_prompt": "你是严谨、可解释的任务执行智能体。使用上游结果完成当前子任务，并明确依据与不确定性。",
                    "prompt_template": f"任务：{subtask.description}\n用户输入：{{input}}\n上游结果：{{upstream}}",
                    "temperature": .2,
                })
            elif chosen.subject_type.value == "ml":
                config.update({
                    "runtime": "python",
                    "requirements": ["numpy", "scikit-learn"],
                    "code": "def run(payload, upstream):\n    # 在独立导出包中加载模型并替换这里的实现\n    values = payload.get('features', [])\n    return {'prediction': None, 'samples': len(values), 'upstream': upstream}\n",
                })
            elif chosen.subject_type.value == "human":
                config.update({
                    "instruction": f"请对“{subtask.description}”进行专业判断，指出需要修订的内容并给出理由。",
                    "approval_criteria": "准确、可解释、符合领域规范，并明确最终责任。",
                })
            else:
                config.update({"connector": "passthrough", "operation": subtask.task_type})
            nodes.append(WorkflowNode(id=node_id, subtask_id=subtask.id, subject_id=chosen.subject_id, subject_type=chosen.subject_type, label=subtask.name, match_score=chosen.score, config=config, explainability={"reason": chosen.explanation, "similarity": chosen.similarity, "alternatives": chosen.alternatives}))
            trace.append({"subtask_id": subtask.id, "selected": chosen.subject_id, "subject_type": chosen.subject_type, "score": chosen.score, "reason": chosen.explanation, "alternatives": chosen.alternatives})

        for subtask in request.task_graph.subtasks:
            for dependency in subtask.dependencies:
                if dependency in subtask_to_node:
                    edges.append(WorkflowEdge(source=subtask_to_node[dependency], target=subtask_to_node[subtask.id]))

        return WorkflowPlan(id=str(uuid4()), task_id=request.task_graph.task_id, nodes=nodes, edges=edges, decision_trace=trace, estimated_cost=round(total_cost, 4), requires_human=any(n.subject_type.value == "human" for n in nodes))
