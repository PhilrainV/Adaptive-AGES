from app.schemas.domain import CapabilitySubject, PlanRequest, SubjectType
from app.services.adaptive_planner import AdaptivePlanner
from app.services.task_understanding import TaskUnderstandingEngine


def subjects():
    return [
        CapabilitySubject(id="llm", name="Reasoner", subject_type=SubjectType.LLM, capability={"reasoning": .96, "generation": .94, "interpretation": .9, "domain_knowledge": .7}, reliability=.9, cost=.35, latency=.3),
        CapabilitySubject(id="xgb", name="XGBoost", subject_type=SubjectType.ML, capability={"prediction": .97, "data_processing": .72, "interpretation": .75}, reliability=.95, cost=.08, latency=.05),
        CapabilitySubject(id="teacher", name="Teacher", subject_type=SubjectType.HUMAN, capability={"human_judgement": .98, "domain_knowledge": .96, "interpretation": .88}, reliability=.92, cost=.7, latency=.8),
        CapabilitySubject(id="etl", name="Data Tool", subject_type=SubjectType.TOOL, capability={"data_processing": .98, "prediction": .2}, reliability=.98, cost=.03, latency=.04),
    ]


def test_end_to_end_planning_selects_heterogeneous_subjects():
    graph = TaskUnderstandingEngine().understand("分析学生数据，预测风险，生成教学建议，最后由教师复核")
    plan = AdaptivePlanner().plan(PlanRequest(task_graph=graph, capability_space=subjects()))
    kinds = {node.subject_type for node in plan.nodes}
    assert SubjectType.ML in kinds
    assert SubjectType.LLM in kinds
    assert SubjectType.HUMAN in kinds
    assert plan.requires_human is True
    assert len(plan.decision_trace) == len(graph.subtasks)


def test_task_graph_dependencies_are_acyclic_chain():
    graph = TaskUnderstandingEngine().understand("读取成绩数据并预测风险，生成报告")
    seen = set()
    for subtask in graph.subtasks:
        assert all(dependency in seen for dependency in subtask.dependencies)
        seen.add(subtask.id)
