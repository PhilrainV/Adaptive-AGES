import io
import zipfile

import pytest

from app.executors.registry import ExecutorRegistry
from app.schemas.domain import SubjectType, WorkflowEdge, WorkflowNode, WorkflowPlan
from app.services.human_assessment import HumanCapabilityAssessmentService
from app.services.workflow_export import export_workflow_bundle
from app.workflow.langgraph_engine import LangGraphExecutionEngine


def sample_plan() -> WorkflowPlan:
    return WorkflowPlan(
        id="workflow-1", task_id="task-1", estimated_cost=.1, requires_human=False, decision_trace=[],
        nodes=[
            WorkflowNode(id="prepare", subtask_id="prepare", subject_id="tool", subject_type=SubjectType.TOOL, label="准备", match_score=.9, config={"connector":"passthrough"}),
            WorkflowNode(id="predict", subtask_id="predict", subject_id="ml", subject_type=SubjectType.ML, label="预测", match_score=.9, config={"code":"def run(payload, upstream):\n    return {'ok': True}\n"}),
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


def test_export_contains_runner_graph_and_editable_ml_module():
    plan = sample_plan().model_dump(mode="json")
    content = export_workflow_bundle(plan, "Test Workflow")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = set(archive.namelist())
        assert {"workflow.json", "run_workflow.py", "input.json", "nodes/predict.py"} <= names
        assert b"def run" in archive.read("nodes/predict.py")


@pytest.mark.asyncio
async def test_langgraph_execution_finishes_without_external_model():
    engine = LangGraphExecutionEngine(ExecutorRegistry())
    result = await engine.execute(sample_plan(), "execution-1", {"features":[[1, 2]]})
    assert result["status"] == "completed"
    assert set(result["node_outputs"]) == {"prepare", "predict"}
