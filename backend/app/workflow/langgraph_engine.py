from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.executors.base import ExecutionContext, HumanInputRequired
from app.executors.registry import ExecutorRegistry
from app.schemas.domain import EdgeType, WorkflowEdge, WorkflowPlan


class WorkflowState(TypedDict, total=False):
    execution_id: str
    payload: dict[str, Any]
    node_outputs: dict[str, Any]
    status: str
    waiting_node: str
    iteration_counts: dict[str, int]
    route_decisions: dict[str, str]


class LangGraphExecutionEngine:
    """Compile DAGs, fan-outs, conditional branches and bounded feedback loops."""

    def __init__(self, registry: ExecutorRegistry | None = None):
        self.registry = registry or ExecutorRegistry()

    def compile(self, plan: WorkflowPlan):
        graph = StateGraph(WorkflowState)
        predecessors = {node.id: [] for node in plan.nodes}
        outgoing: dict[str, list[WorkflowEdge]] = {node.id: [] for node in plan.nodes}
        for edge in plan.edges:
            predecessors[edge.target].append(edge.source)
            outgoing[edge.source].append(edge)
        controlled_sources = {
            source
            for source, edges in outgoing.items()
            if any(edge.edge_type != EdgeType.DEFAULT or edge.condition for edge in edges)
        }

        for node in plan.nodes:
            async def run_node(state: WorkflowState, current=node) -> WorkflowState:
                if state.get("status") == "waiting_for_human":
                    return state
                inputs = {
                    "input": state.get("payload", {}),
                    "upstream": {
                        key: state.get("node_outputs", {}).get(key)
                        for key in predecessors[current.id]
                    },
                }
                context = ExecutionContext(
                    execution_id=state["execution_id"],
                    node_id=current.id,
                    subject_id=current.subject_id,
                    config=current.config,
                )
                try:
                    output = await self.registry.get(current.subject_type.value).execute(inputs, context)
                    next_state: WorkflowState = {
                        **state,
                        "node_outputs": {**state.get("node_outputs", {}), current.id: output},
                        "status": "running",
                    }
                    if current.id in controlled_sources:
                        decision, counts = self._select_route(current.id, output, outgoing[current.id], state)
                        next_state["iteration_counts"] = counts
                        next_state["route_decisions"] = {
                            **state.get("route_decisions", {}),
                            current.id: decision,
                        }
                    return next_state
                except HumanInputRequired as pause:
                    return {
                        **state,
                        "status": "waiting_for_human",
                        "waiting_node": current.id,
                        "node_outputs": {**state.get("node_outputs", {}), current.id: pause.payload},
                    }

            graph.add_node(node.id, run_node)

        roots = [node.id for node in plan.nodes if not predecessors[node.id]]
        if not roots:
            raise ValueError("workflow has no entry node; check loop dependencies")
        if len(roots) != 1:
            graph.add_node("__start_router__", lambda state: state)
            graph.set_entry_point("__start_router__")
            for root in roots:
                graph.add_edge("__start_router__", root)
        else:
            graph.set_entry_point(roots[0])

        for node in plan.nodes:
            edges = outgoing[node.id]
            if not edges:
                graph.add_edge(node.id, END)
            elif node.id in controlled_sources:
                destinations = {edge.target: edge.target for edge in edges}
                destinations["__end__"] = END

                def route(state: WorkflowState, source=node.id) -> str:
                    return state.get("route_decisions", {}).get(source, "__end__")

                graph.add_conditional_edges(node.id, route, destinations)
            else:
                for edge in edges:
                    graph.add_edge(node.id, edge.target)
        return graph.compile()

    async def execute(self, plan: WorkflowPlan, execution_id: str, payload: dict[str, Any]) -> WorkflowState:
        graph = self.compile(plan)
        result = await graph.ainvoke(
            {
                "execution_id": execution_id,
                "payload": payload,
                "node_outputs": {},
                "status": "running",
                "iteration_counts": {},
                "route_decisions": {},
            },
            {"recursion_limit": 100},
        )
        if result.get("status") == "running":
            result["status"] = "completed"
        return result

    def _select_route(
        self,
        source: str,
        output: Any,
        edges: list[WorkflowEdge],
        state: WorkflowState,
    ) -> tuple[str, dict[str, int]]:
        counts = dict(state.get("iteration_counts", {}))
        for edge in edges:
            if edge.edge_type != EdgeType.LOOP:
                continue
            key = f"{source}->{edge.target}"
            if counts.get(key, 0) < edge.max_iterations and self._condition_matches(edge.condition, output, state):
                counts[key] = counts.get(key, 0) + 1
                return edge.target, counts
        for edge in edges:
            if (
                edge.edge_type == EdgeType.CONDITIONAL
                and edge.condition not in {None, "", "else"}
                and self._condition_matches(edge.condition, output, state)
            ):
                return edge.target, counts
        fallback = next(
            (
                edge
                for edge in edges
                if edge.edge_type == EdgeType.DEFAULT or edge.condition in {None, "", "else"}
            ),
            None,
        )
        return (fallback.target if fallback else "__end__"), counts

    @classmethod
    def _condition_matches(cls, expression: str | None, output: Any, state: WorkflowState) -> bool:
        if not expression or expression == "else":
            return False
        match = re.fullmatch(r"\s*([A-Za-z_][\w.]*)\s*(==|!=|>=|<=|>|<)?\s*(.*?)\s*", expression)
        if not match:
            return False
        path, operator, raw_expected = match.groups()
        actual = cls._resolve_path(output, path)
        if actual is None and path.startswith("payload."):
            actual = cls._resolve_path(state.get("payload", {}), path.removeprefix("payload."))
        if not operator:
            return bool(actual)
        expected = cls._parse_literal(raw_expected)
        try:
            return {
                "==": actual == expected,
                "!=": actual != expected,
                ">=": actual >= expected,
                "<=": actual <= expected,
                ">": actual > expected,
                "<": actual < expected,
            }[operator]
        except TypeError:
            return False

    @staticmethod
    def _resolve_path(value: Any, path: str) -> Any:
        current = value
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _parse_literal(value: str) -> Any:
        stripped = value.strip().strip("\"'")
        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            return float(stripped) if "." in stripped else int(stripped)
        except ValueError:
            return stripped
