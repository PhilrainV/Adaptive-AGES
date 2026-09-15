import io
import zipfile

import pytest

from app.agents.ability_diagnosis_agent import AbilityDiagnosisAgent
from app.agents.capability_planning_agent import CapabilityPlanningAgent
from app.agents.problem_analysis_agent import ProblemAnalysisAgent
from app.agents.test_generation_agent import (
    TestGenerationAgent as AssessmentTestGenerationAgent,
)
from app.executors.registry import ExecutorRegistry
from app.schemas.domain import (
    CapabilitySubject,
    PlanningCreateWorkflowRequest,
    PlanningDiagnoseRequest,
    PlanningStartRequest,
    SubjectType,
    WorkflowEdge,
    WorkflowNode,
    WorkflowPlan,
)
from app.services.human_assessment import HumanCapabilityAssessmentService
from app.services.workflow_export import export_workflow_bundle
from app.workflow.langgraph_engine import LangGraphExecutionEngine


def sample_plan() -> WorkflowPlan:
    return WorkflowPlan(
        id="workflow-1",
        task_id="task-1",
        estimated_cost=.1,
        requires_human=False,
        decision_trace=[],
        nodes=[
            WorkflowNode(
                id="prepare",
                subtask_id="prepare",
                subject_id="tool",
                subject_type=SubjectType.TOOL,
                label="准备",
                match_score=.9,
                config={"connector": "passthrough"},
            ),
            WorkflowNode(
                id="predict",
                subtask_id="predict",
                subject_id="ml",
                subject_type=SubjectType.ML,
                label="预测",
                match_score=.9,
                config={
                    "code": "def run(payload, upstream):\n    return {'ok': True}\n"
                },
            ),
        ],
        edges=[WorkflowEdge(source="prepare", target="predict")],
    )


def test_task_adaptive_assessment_updates_multiple_dimensions():
    service = HumanCapabilityAssessmentService()
    questions = service.generate("构建一个教育数据预测与教师复核系统")
    answers = {item["id"]: item["correct_index"] for item in questions}
    result = service.score(questions, answers)
    assert result["overall"] == 1
    assert set(result["capability"]) == set(service.dimensions)
    assert "教育数据预测" in questions[0]["prompt"]


def test_diagnosis_and_planning_requests_are_separate_stages():
    diagnosis_request = PlanningDiagnoseRequest(answers={"question-1": 2})
    planning_request = PlanningCreateWorkflowRequest(capability_space=[])

    assert diagnosis_request.model_dump() == {"answers": {"question-1": 2}}
    assert "capability_space" not in PlanningDiagnoseRequest.model_fields
    assert "answers" not in PlanningCreateWorkflowRequest.model_fields
    assert planning_request.capability_space == []


def test_planning_start_supports_assessment_and_direct_modes():
    assessed = PlanningStartRequest(prompt="分析问题并在测试后规划")
    direct = PlanningStartRequest(
        prompt="分析问题后直接生成规划",
        assessment_enabled=False,
        capability_space=[
            CapabilitySubject(
                id="llm",
                name="LLM",
                subject_type=SubjectType.LLM,
                capability={"reasoning": .9, "generation": .9},
            )
        ],
    )

    assert assessed.assessment_enabled is True
    assert direct.assessment_enabled is False
    assert direct.capability_space[0].id == "llm"


@pytest.mark.asyncio
async def test_four_planning_agents_form_an_assessment_first_pipeline():
    graph = await ProblemAnalysisAgent().run("分析学生数据，预测风险并由教师复核")
    questions, mode = await AssessmentTestGenerationAgent().run(graph)
    answers = {item["id"]: item["correct_index"] for item in questions}
    diagnosis = AbilityDiagnosisAgent().run(questions, answers, graph)
    subjects = [
        CapabilitySubject(
            id="llm",
            name="LLM",
            subject_type=SubjectType.LLM,
            capability={"reasoning": .95, "generation": .95, "interpretation": .9},
        ),
        CapabilitySubject(
            id="ml",
            name="ML",
            subject_type=SubjectType.ML,
            capability={"prediction": .96, "data_processing": .9},
        ),
        CapabilitySubject(
            id="human",
            name="Human",
            subject_type=SubjectType.HUMAN,
            capability={"human_judgement": .1, "domain_knowledge": .1},
            reliability=.9,
        ),
        CapabilitySubject(
            id="tool",
            name="Tool",
            subject_type=SubjectType.TOOL,
            capability={"data_processing": .98},
        ),
    ]
    plan = CapabilityPlanningAgent().run(graph, subjects, diagnosis)

    assert mode == "dina_rule"
    assert diagnosis["method"] == "bayesian_dina"
    assert diagnosis["overall"] > .85
    assert diagnosis["planning_capability"]["human_judgement"] > .85
    assert plan.task_id == graph.task_id
    assert plan.decision_trace[0]["agent"] == "capability_planning_agent"


def test_export_contains_runner_graph_and_editable_ml_module():
    plan = sample_plan().model_dump(mode="json")
    content = export_workflow_bundle(plan, "Test Workflow")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert {
            "workflow.json",
            "run_workflow.py",
            "input.json",
            "nodes/predict.py",
        } <= names
        assert b"def run" in archive.read("nodes/predict.py")


@pytest.mark.asyncio
async def test_langgraph_execution_finishes_without_external_model():
    engine = LangGraphExecutionEngine(ExecutorRegistry())
    result = await engine.execute(sample_plan(), "execution-1", {"features": [[1, 2]]})
    assert result["status"] == "completed"
    assert set(result["node_outputs"]) == {"prepare", "predict"}
