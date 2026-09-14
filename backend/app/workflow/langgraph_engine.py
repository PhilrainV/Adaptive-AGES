from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.executors.base import ExecutionContext, HumanInputRequired
from app.executors.registry import ExecutorRegistry
from app.schemas.domain import WorkflowPlan


class WorkflowState(TypedDict, total=False):
    execution_id: str
    payload: dict[str, Any]
    node_outputs: dict[str, Any]
    status: str
    waiting_node: str


class LangGraphExecutionEngine:
    def __init__(self, registry: ExecutorRegistry | None = None):
        self.registry = registry or ExecutorRegistry()

    def compile(self, plan: WorkflowPlan):
        graph = StateGraph(WorkflowState)
        predecessors = {node.id: [] for node in plan.nodes}
        successors = {node.id: [] for node in plan.nodes}
        for edge in plan.edges:
            predecessors[edge.target].append(edge.source)
            successors[edge.source].append(edge.target)

        for node in plan.nodes:
            async def run_node(state: WorkflowState, current=node) -> WorkflowState:
                if state.get("status") == "waiting_for_human":
                    return state
                inputs = {"input": state.get("payload", {}), "upstream": {key: state.get("node_outputs", {}).get(key) for key in predecessors[current.id]}}
                context = ExecutionContext(execution_id=state["execution_id"], node_id=current.id, subject_id=current.subject_id, config=current.config)
                try:
                    output = await self.registry.get(current.subject_type.value).execute(inputs, context)
                    outputs = {**state.get("node_outputs", {}), current.id: output}
                    return {**state, "node_outputs": outputs, "status": "running"}
                except HumanInputRequired as pause:
                    return {**state, "status": "waiting_for_human", "waiting_node": current.id, "node_outputs": {**state.get("node_outputs", {}), current.id: pause.payload}}
            graph.add_node(node.id, run_node)

        roots = [node.id for node in plan.nodes if not predecessors[node.id]]
        if len(roots) != 1:
            graph.add_node("__start_router__", lambda state: state)
            graph.set_entry_point("__start_router__")
            for root in roots:
                graph.add_edge("__start_router__", root)
        else:
            graph.set_entry_point(roots[0])
        for node in plan.nodes:
            if successors[node.id]:
                for target in successors[node.id]: graph.add_edge(node.id, target)
            else: graph.add_edge(node.id, END)
        return graph.compile()

    async def execute(self, plan: WorkflowPlan, execution_id: str, payload: dict[str, Any]) -> WorkflowState:
        graph = self.compile(plan)
        result = await graph.ainvoke({"execution_id": execution_id, "payload": payload, "node_outputs": {}, "status": "running"})
        if result.get("status") == "running":
            result["status"] = "completed"
        return result
