from fastapi import APIRouter, HTTPException, Response, status
from langchain_openai import ChatOpenAI
from sqlalchemy import select

from app.api.deps import CurrentUserId, DbSession
from app.core.secrets import decrypt_secret, encrypt_secret
from app.core.security import create_access_token, hash_password, verify_password
from app.executors.registry import ExecutorRegistry
from app.models.entities import (
    Agent,
    AgentCapability,
    Feedback,
    HumanAssessment,
    HumanProfile,
    ModelSetting,
    Task,
    User,
    Workflow,
    WorkflowExecution,
)
from app.models.entities import (
    TaskGraph as TaskGraphRecord,
)
from app.schemas.domain import (
    AgentCreate,
    FeedbackCreate,
    HumanAssessmentGenerateRequest,
    HumanAssessmentSubmitRequest,
    LoginRequest,
    ModelSettingsUpdate,
    PlanRequest,
    TaskUnderstandRequest,
    UserCreate,
    WorkflowPlan,
    WorkflowRunRequest,
    WorkflowUpdate,
)
from app.services.adaptive_planner import AdaptivePlanner
from app.services.human_assessment import HumanCapabilityAssessmentService
from app.services.task_understanding import TaskUnderstandingEngine
from app.services.workflow_export import export_workflow_bundle
from app.workflow.langgraph_engine import LangGraphExecutionEngine

router = APIRouter()
understanding = TaskUnderstandingEngine()
planner = AdaptivePlanner()
assessment = HumanCapabilityAssessmentService()


async def model_config_for(db: DbSession, user_id: str) -> dict:
    record = await db.scalar(select(ModelSetting).where(ModelSetting.user_id == user_id))
    if not record:
        return {}
    try:
        api_key = decrypt_secret(record.api_key_encrypted)
    except Exception:  # noqa: BLE001 - a rotated secret invalidates earlier ciphertext
        api_key = None
    return {
        "provider": record.provider, "model": record.model, "base_url": record.base_url,
        "api_key": api_key, "temperature": record.temperature,
    }


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DbSession):
    if await db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=payload.email, display_name=payload.display_name,
        password_hash=hash_password(payload.password),
        adaptive_profile={"programming": .5, "ai_literacy": .5, "domain_knowledge": .5},
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {
        "id": user.id, "email": user.email, "display_name": user.display_name,
        "access_token": create_access_token(user.id), "token_type": "bearer",
    }


@router.post("/auth/login")
async def login(payload: LoginRequest, db: DbSession):
    user = await db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": create_access_token(user.id), "token_type": "bearer"}


@router.get("/agents")
async def list_agents(user_id: CurrentUserId, db: DbSession):
    agents = (await db.scalars(select(Agent).where(Agent.owner_id == user_id))).all()
    return [{
        "id": item.id, "name": item.name, "agent_type": item.agent_type,
        "description": item.description, "config": item.config, "is_active": item.is_active,
    } for item in agents]


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreate, user_id: CurrentUserId, db: DbSession):
    agent = Agent(
        owner_id=user_id, name=payload.name, agent_type=payload.agent_type.value,
        description=payload.description, endpoint=payload.endpoint, config=payload.config,
    )
    for dimension, score in payload.capabilities.items():
        agent.capabilities.append(AgentCapability(dimension=dimension, score=score, confidence=.6))
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return {"id": agent.id, "name": agent.name, "agent_type": agent.agent_type}


@router.post("/tasks/understand")
async def understand_task(payload: TaskUnderstandRequest, user_id: CurrentUserId, db: DbSession):
    task = Task(owner_id=user_id, title=payload.prompt[:120], prompt=payload.prompt, constraints=payload.constraints)
    db.add(task)
    await db.flush()
    graph = await understanding.understand_async(payload.prompt, await model_config_for(db, user_id))
    graph.task_id = task.id
    task.complexity = graph.complexity
    task.task_type = graph.subtasks[-1].task_type
    db.add(TaskGraphRecord(task_id=task.id, graph=graph.model_dump(mode="json")))
    await db.commit()
    return graph


@router.post("/workflows/plan", response_model=WorkflowPlan)
async def plan_workflow(payload: PlanRequest, user_id: CurrentUserId, db: DbSession):
    task = await db.scalar(select(Task).where(Task.id == payload.task_graph.task_id, Task.owner_id == user_id))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    plan = planner.plan(payload)
    record = Workflow(
        owner_id=user_id, task_id=task.id, name=f"{task.title} · 自适应工作流",
        definition=plan.model_dump(mode="json"), decision_trace={"items": plan.decision_trace}, status="ready",
    )
    db.add(record)
    await db.flush()
    plan.id = record.id
    record.definition = plan.model_dump(mode="json")
    await db.commit()
    return plan


@router.put("/workflows/{workflow_id}", response_model=WorkflowPlan)
async def update_workflow(workflow_id: str, payload: WorkflowUpdate, user_id: CurrentUserId, db: DbSession):
    workflow = await db.scalar(select(Workflow).where(Workflow.id == workflow_id, Workflow.owner_id == user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    definition = dict(workflow.definition)
    definition["nodes"] = [node.model_dump(mode="json") for node in payload.nodes]
    definition["edges"] = [edge.model_dump(mode="json") for edge in payload.edges]
    definition["requires_human"] = any(node.subject_type.value == "human" for node in payload.nodes)
    workflow.definition = definition
    workflow.version += 1
    if payload.name:
        workflow.name = payload.name
    await db.commit()
    return WorkflowPlan.model_validate(definition)


@router.post("/workflows/{workflow_id}/execute")
async def execute_saved_workflow(workflow_id: str, payload: WorkflowRunRequest, user_id: CurrentUserId, db: DbSession):
    workflow = await db.scalar(select(Workflow).where(Workflow.id == workflow_id, Workflow.owner_id == user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    plan = WorkflowPlan.model_validate(workflow.definition)
    execution = WorkflowExecution(workflow_id=workflow.id, status="running", input=payload.input)
    db.add(execution)
    await db.flush()
    engine = LangGraphExecutionEngine(ExecutorRegistry(await model_config_for(db, user_id)))
    try:
        result = await engine.execute(plan, execution.id, payload.input)
    except Exception as exc:
        execution.status = "failed"
        execution.output = {"error": str(exc)}
        await db.commit()
        raise HTTPException(status_code=502, detail=f"Workflow execution failed: {exc}") from exc
    execution.status = result.get("status", "completed")
    execution.state = dict(result)
    execution.output = result.get("node_outputs", {})
    await db.commit()
    return {"execution_id": execution.id, **result}


@router.post("/workflows/execute")
async def execute_workflow(plan: WorkflowPlan, user_id: CurrentUserId, db: DbSession):
    workflow = await db.scalar(select(Workflow).where(Workflow.id == plan.id, Workflow.owner_id == user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    execution = WorkflowExecution(workflow_id=workflow.id, status="running", input={})
    db.add(execution)
    await db.flush()
    engine = LangGraphExecutionEngine(ExecutorRegistry(await model_config_for(db, user_id)))
    result = await engine.execute(plan, execution.id, {})
    execution.status = result.get("status", "completed")
    execution.state = dict(result)
    execution.output = result.get("node_outputs", {})
    await db.commit()
    return {"execution_id": execution.id, **result}


@router.get("/workflows/{workflow_id}/export")
async def export_workflow(workflow_id: str, user_id: CurrentUserId, db: DbSession):
    workflow = await db.scalar(select(Workflow).where(Workflow.id == workflow_id, Workflow.owner_id == user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    content = export_workflow_bundle(workflow.definition, workflow.name)
    return Response(
        content=content, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="adaptive-workflow-{workflow.id[:8]}.zip"'},
    )


@router.get("/settings/model")
async def get_model_settings(user_id: CurrentUserId, db: DbSession):
    config = await model_config_for(db, user_id)
    return {
        "provider": config.get("provider", "openai-compatible"),
        "model": config.get("model", "gpt-4.1-mini"), "base_url": config.get("base_url"),
        "temperature": config.get("temperature", .2), "api_key_configured": bool(config.get("api_key")),
    }


@router.put("/settings/model")
async def save_model_settings(payload: ModelSettingsUpdate, user_id: CurrentUserId, db: DbSession):
    record = await db.scalar(select(ModelSetting).where(ModelSetting.user_id == user_id))
    if not record:
        record = ModelSetting(user_id=user_id)
        db.add(record)
    record.provider = payload.provider
    record.model = payload.model
    record.base_url = payload.base_url or None
    record.temperature = payload.temperature
    if payload.api_key:
        record.api_key_encrypted = encrypt_secret(payload.api_key)
    await db.commit()
    return {"saved": True, "api_key_configured": bool(record.api_key_encrypted)}


@router.post("/settings/model/test")
async def test_model_settings(user_id: CurrentUserId, db: DbSession):
    config = await model_config_for(db, user_id)
    if not config.get("api_key"):
        raise HTTPException(status_code=400, detail="请先保存 API Key")
    try:
        client = ChatOpenAI(
            api_key=config["api_key"], base_url=config.get("base_url") or None,
            model=config.get("model") or "gpt-4.1-mini", temperature=0, timeout=30,
        )
        response = await client.ainvoke("仅回复 OK")
        return {"ok": True, "message": str(response.content)[:120]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"模型连接失败：{exc}") from exc


@router.get("/human-assessments/profile")
async def get_human_profile(user_id: CurrentUserId, db: DbSession):
    profile = await db.scalar(select(HumanProfile).where(HumanProfile.user_id == user_id))
    return {"capability": profile.capability_vector if profile else {}, "evidence": profile.decision_history if profile else {}}


@router.post("/human-assessments/generate", status_code=status.HTTP_201_CREATED)
async def generate_human_assessment(payload: HumanAssessmentGenerateRequest, user_id: CurrentUserId, db: DbSession):
    questions = assessment.generate(payload.design_requirement)
    record = HumanAssessment(user_id=user_id, design_requirement=payload.design_requirement, questions=questions)
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return {
        "assessment_id": record.id, "design_requirement": record.design_requirement,
        "questions": assessment.public_questions(questions),
    }


@router.post("/human-assessments/{assessment_id}/submit")
async def submit_human_assessment(assessment_id: str, payload: HumanAssessmentSubmitRequest, user_id: CurrentUserId, db: DbSession):
    record = await db.scalar(select(HumanAssessment).where(HumanAssessment.id == assessment_id, HumanAssessment.user_id == user_id))
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")
    result = assessment.score(record.questions, payload.answers)
    record.answers = payload.answers
    record.result = result
    record.status = "completed"
    profile = await db.scalar(select(HumanProfile).where(HumanProfile.user_id == user_id))
    if not profile:
        profile = HumanProfile(user_id=user_id)
        db.add(profile)
    profile.capability_vector = result["capability"]
    profile.decision_history = {
        "source": "task_adaptive_assessment", "assessment_id": record.id,
        "design_requirement": record.design_requirement, "overall": result["overall"],
    }
    user = await db.scalar(select(User).where(User.id == user_id))
    if user:
        user.adaptive_profile = result["capability"]
    await db.commit()
    return result


@router.post("/executions/{execution_id}/human-feedback")
async def submit_human_feedback(execution_id: str, payload: FeedbackCreate, user_id: CurrentUserId, db: DbSession):
    execution = await db.scalar(
        select(WorkflowExecution).join(Workflow).where(
            WorkflowExecution.id == execution_id, Workflow.owner_id == user_id,
        )
    )
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    record = Feedback(
        execution_id=execution.id, user_id=user_id, node_id=payload.node_id,
        feedback_type=payload.feedback_type, score=payload.score,
        correction=payload.correction, rationale=payload.rationale,
    )
    db.add(record)
    execution.status = "resumable"
    await db.commit()
    return {"execution_id": execution_id, "status": "resumable", "feedback_id": record.id}
