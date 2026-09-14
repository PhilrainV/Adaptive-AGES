from copy import deepcopy
from datetime import datetime, timezone

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
    NodeModelSettingsUpdate,
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


PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "bailian-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "bailian-intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


def normalized_base_url(provider: str, base_url: str | None) -> str | None:
    value = (base_url or PROVIDER_BASE_URLS.get(provider) or "").strip().rstrip("/")
    if not value:
        if provider in {"newapi", "local"}:
            raise HTTPException(status_code=400, detail="该接口类型必须填写实际 Base URL（通常以 /v1 结尾）")
        return None
    if not value.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Base URL 必须以 http:// 或 https:// 开头")
    return value


def public_workflow_definition(definition: dict) -> dict:
    result = deepcopy(definition)
    for node in result.get("nodes", []):
        config = node.get("config") or {}
        configured = bool(config.pop("api_key_encrypted", None))
        if configured:
            config["api_key_configured"] = True
        node["config"] = config
    return result


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


@router.get("/dashboard")
async def get_dashboard(user_id: CurrentUserId, db: DbSession):
    user = await db.scalar(select(User).where(User.id == user_id))
    agents = list((await db.scalars(select(Agent).where(Agent.owner_id == user_id))).all())
    workflows = list((await db.scalars(
        select(Workflow).where(Workflow.owner_id == user_id).order_by(Workflow.updated_at.desc())
    )).all())
    tasks = list((await db.scalars(select(Task).where(Task.owner_id == user_id))).all())
    executions = list((await db.scalars(
        select(WorkflowExecution).join(Workflow).where(Workflow.owner_id == user_id)
        .order_by(WorkflowExecution.created_at.desc())
    )).all())
    task_map = {item.id: item for item in tasks}
    workflow_map = {item.id: item for item in workflows}
    latest_execution = {}
    for item in executions:
        latest_execution.setdefault(item.workflow_id, item)
    terminal = [item for item in executions if item.status in {"completed", "failed"}]
    completed = sum(item.status == "completed" for item in terminal)
    now = datetime.now(timezone.utc)
    month_executions = sum(
        bool(item.created_at and item.created_at.year == now.year and item.created_at.month == now.month)
        for item in executions
    )
    recent_tasks = []
    for workflow in workflows[:8]:
        execution = latest_execution.get(workflow.id)
        nodes = workflow.definition.get("nodes", [])
        types = list(dict.fromkeys(node.get("subject_type", "tool") for node in nodes))
        scores = [float(node.get("match_score", 0)) for node in nodes if node.get("match_score") is not None]
        recent_tasks.append({
            "id": workflow.id,
            "title": task_map.get(workflow.task_id).title if task_map.get(workflow.task_id) else workflow.name,
            "agents": types,
            "status": execution.status if execution else workflow.status,
            "match_score": sum(scores) / len(scores) if scores else 0,
            "updated_at": (execution.created_at if execution else workflow.updated_at).isoformat(),
        })
    activities = []
    for execution in executions[:8]:
        workflow = workflow_map.get(execution.workflow_id)
        activities.append({
            "id": execution.id,
            "title": workflow.name if workflow else "工作流执行",
            "status": execution.status,
            "created_at": (execution.created_at or now).isoformat(),
        })
    return {
        "display_name": user.display_name if user else "用户",
        "metrics": {
            "agents": len(agents),
            "workflows": len(workflows),
            "month_executions": month_executions,
            "success_rate": completed / len(terminal) if terminal else 0,
            "running": sum(item.status in {"queued", "running"} for item in executions),
            "waiting_human": sum(item.status == "waiting_for_human" for item in executions),
            "failed": sum(item.status == "failed" for item in executions),
        },
        "recent_tasks": recent_tasks,
        "activities": activities,
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
    previous_nodes = {item.get("id"): item for item in definition.get("nodes", [])}
    next_nodes = []
    for node in payload.nodes:
        item = node.model_dump(mode="json")
        old_config = (previous_nodes.get(item["id"], {}).get("config") or {})
        if old_config.get("api_key_encrypted") and not item["config"].get("use_default_model", True):
            item["config"]["api_key_encrypted"] = old_config["api_key_encrypted"]
            item["config"]["api_key_configured"] = True
        item["config"].pop("api_key", None)
        next_nodes.append(item)
    definition["nodes"] = next_nodes
    definition["edges"] = [edge.model_dump(mode="json") for edge in payload.edges]
    definition["requires_human"] = any(node.subject_type.value == "human" for node in payload.nodes)
    workflow.definition = definition
    workflow.version += 1
    if payload.name:
        workflow.name = payload.name
    await db.commit()
    return WorkflowPlan.model_validate(public_workflow_definition(definition))


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
    record.model = payload.model.strip()
    record.base_url = normalized_base_url(payload.provider, payload.base_url)
    record.temperature = payload.temperature
    if payload.api_key:
        record.api_key_encrypted = encrypt_secret(payload.api_key)
    await db.commit()
    return {"saved": True, "api_key_configured": bool(record.api_key_encrypted)}


@router.put("/workflows/{workflow_id}/nodes/{node_id}/model-settings")
async def save_node_model_settings(
    workflow_id: str,
    node_id: str,
    payload: NodeModelSettingsUpdate,
    user_id: CurrentUserId,
    db: DbSession,
):
    workflow = await db.scalar(select(Workflow).where(Workflow.id == workflow_id, Workflow.owner_id == user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    definition = deepcopy(workflow.definition)
    node = next((item for item in definition.get("nodes", []) if item.get("id") == node_id), None)
    if not node:
        raise HTTPException(status_code=404, detail="Workflow node not found")
    if node.get("subject_type") != "llm":
        raise HTTPException(status_code=400, detail="只有 LLM 节点可以配置独立模型")
    config = dict(node.get("config") or {})
    config["use_default_model"] = payload.use_default
    config["modality"] = payload.modality
    config["temperature"] = payload.temperature
    if payload.use_default:
        for key in ("provider", "model", "base_url", "api_key_encrypted", "api_key_configured"):
            config.pop(key, None)
    else:
        config["provider"] = payload.provider
        config["model"] = payload.model.strip()
        config["base_url"] = normalized_base_url(payload.provider, payload.base_url)
        if payload.api_key:
            config["api_key_encrypted"] = encrypt_secret(payload.api_key)
        if not config.get("api_key_encrypted"):
            raise HTTPException(status_code=400, detail="独立模型需要填写并保存 API Key")
        config["api_key_configured"] = True
    config.pop("api_key", None)
    node["config"] = config
    workflow.definition = definition
    workflow.version += 1
    await db.commit()
    return {
        "saved": True,
        "use_default": payload.use_default,
        "api_key_configured": bool(config.get("api_key_encrypted")),
    }


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
