from copy import deepcopy
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Response, status
from langchain_openai import ChatOpenAI
from sqlalchemy import select

from app.agents import (
    AbilityDiagnosisAgent,
    CapabilityPlanningAgent,
    ProblemAnalysisAgent,
    TestGenerationAgent,
)
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
    LoginRequest,
    ModelSettingsUpdate,
    NodeModelSettingsUpdate,
    PlanningCreateWorkflowRequest,
    PlanningDiagnoseRequest,
    PlanningStartRequest,
    PlanRequest,
    TaskGraph,
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
problem_analysis_agent = ProblemAnalysisAgent(understanding)
test_generation_agent = TestGenerationAgent(assessment)
ability_diagnosis_agent = AbilityDiagnosisAgent()
capability_planning_agent = CapabilityPlanningAgent()


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
    tasks = list(
        (
            await db.scalars(
                select(Task).where(Task.owner_id == user_id).order_by(Task.updated_at.desc())
            )
        ).all()
    )
    executions = list((await db.scalars(
        select(WorkflowExecution).join(Workflow).where(Workflow.owner_id == user_id)
        .order_by(WorkflowExecution.created_at.desc())
    )).all())
    workflow_map = {item.id: item for item in workflows}
    latest_execution = {}
    for item in executions:
        latest_execution.setdefault(item.workflow_id, item)
    terminal = [item for item in executions if item.status in {"completed", "failed"}]
    completed = sum(item.status == "completed" for item in terminal)
    now = datetime.now(UTC)
    month_executions = sum(
        bool(item.created_at and item.created_at.year == now.year and item.created_at.month == now.month)
        for item in executions
    )
    latest_workflow_by_task = {}
    for workflow in workflows:
        latest_workflow_by_task.setdefault(workflow.task_id, workflow)
    recent_tasks = []
    for task in tasks[:8]:
        workflow = latest_workflow_by_task.get(task.id)
        execution = latest_execution.get(workflow.id) if workflow else None
        nodes = workflow.definition.get("nodes", []) if workflow else []
        types = list(dict.fromkeys(node.get("subject_type", "tool") for node in nodes))
        scores = [float(node.get("match_score", 0)) for node in nodes if node.get("match_score") is not None]
        recent_tasks.append({
            "id": task.id,
            "workflow_id": workflow.id if workflow else None,
            "title": task.title,
            "agents": types,
            "status": execution.status if execution else workflow.status if workflow else task.status,
            "match_score": sum(scores) / len(scores) if scores else 0,
            "updated_at": (
                execution.created_at if execution else workflow.updated_at if workflow else task.updated_at
            ).isoformat(),
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


@router.post("/planning-sessions/start", status_code=status.HTTP_201_CREATED)
async def start_planning_session(
    payload: PlanningStartRequest,
    user_id: CurrentUserId,
    db: DbSession,
):
    """Analyse the problem, then either assess the user or create a direct plan."""
    task = Task(
        owner_id=user_id,
        title=payload.prompt[:120],
        prompt=payload.prompt,
        constraints=payload.constraints,
        status="assessing" if payload.assessment_enabled else "planning",
    )
    db.add(task)
    await db.flush()
    model_config = await model_config_for(db, user_id)
    agent_overrides = {
        name: config.model_dump(mode="json")
        for name, config in payload.agent_overrides.items()
    }
    graph = await problem_analysis_agent.run(
        payload.prompt,
        model_config,
        agent_overrides.get(problem_analysis_agent.name),
    )
    graph.task_id = task.id
    task.complexity = graph.complexity
    task.task_type = graph.subtasks[-1].task_type
    db.add(TaskGraphRecord(task_id=task.id, graph=graph.model_dump(mode="json")))

    if not payload.assessment_enabled:
        if not payload.capability_space:
            raise HTTPException(
                status_code=400,
                detail="直接规划模式必须提供可用主体能力空间",
            )
        diagnosis = {
            "method": "assessment_disabled",
            "personalization_enabled": False,
            "overall": None,
            "confidence": 0,
            "capability": {},
            "planning_capability": {},
            "weakest_dimensions": [],
            "diagnostic_summary": {
                "source": "none",
                "reason": "用户关闭了测试生成和能力诊断；规划不读取用户画像",
            },
        }
        plan = await capability_planning_agent.run_async(
            graph,
            payload.capability_space,
            diagnosis,
            payload.weights,
            model_config,
            agent_overrides.get(capability_planning_agent.name),
        )
        workflow = Workflow(
            owner_id=user_id,
            task_id=task.id,
            name=f"{task.title} · 直接规划工作流",
            definition=plan.model_dump(mode="json"),
            decision_trace={"items": plan.decision_trace},
            status="ready",
        )
        db.add(workflow)
        await db.flush()
        plan.id = workflow.id
        workflow.definition = plan.model_dump(mode="json")
        task.status = "planned"
        await db.commit()
        return {
            "session_id": None,
            "task_id": task.id,
            "status": "completed",
            "assessment_enabled": False,
            "task_graph": graph,
            "questions": [],
            "diagnosis": diagnosis,
            "workflow": plan,
            "agent_trace": [
                {
                    "agent": problem_analysis_agent.name,
                    "status": "completed",
                    "summary": f"已解析为 {len(graph.subtasks)} 个子任务",
                    "mode": graph.planning_mode,
                },
                {
                    "agent": test_generation_agent.name,
                    "status": "skipped",
                    "summary": "用户关闭能力测试",
                },
                {
                    "agent": ability_diagnosis_agent.name,
                    "status": "skipped",
                    "summary": "未读取或更新用户画像",
                },
                {
                    "agent": capability_planning_agent.name,
                    "status": "completed",
                    "summary": f"已直接生成 {len(plan.nodes)} 个协同节点",
                },
            ],
        }

    questions, generation_mode = await test_generation_agent.run(
        graph,
        model_config,
        agent_overrides.get(test_generation_agent.name),
    )
    assessment_record = HumanAssessment(
        user_id=user_id,
        design_requirement=payload.prompt,
        questions=questions,
        result={
            "task_id": task.id,
            "task_graph": graph.model_dump(mode="json"),
            "assessment_enabled": True,
            "test_generation_mode": generation_mode,
            "agent_overrides": agent_overrides,
        },
        status="awaiting_answers",
    )
    db.add(assessment_record)
    await db.commit()
    await db.refresh(assessment_record)
    return {
        "session_id": assessment_record.id,
        "task_id": task.id,
        "status": "awaiting_answers",
        "assessment_enabled": True,
        "task_graph": graph,
        "questions": assessment.public_questions(questions),
        "workflow": None,
        "agent_trace": [
            {
                "agent": problem_analysis_agent.name,
                "status": "completed",
                "summary": f"已解析为 {len(graph.subtasks)} 个子任务",
                "mode": graph.planning_mode,
            },
            {
                "agent": test_generation_agent.name,
                "status": "completed",
                "summary": f"已生成 {len(questions)} 道任务自适应测试题",
                "mode": generation_mode,
            },
            {"agent": ability_diagnosis_agent.name, "status": "waiting_for_answers"},
            {"agent": capability_planning_agent.name, "status": "waiting_for_diagnosis"},
        ],
    }


@router.post("/planning-sessions/{session_id}/diagnose")
async def diagnose_planning_session(
    session_id: str,
    payload: PlanningDiagnoseRequest,
    user_id: CurrentUserId,
    db: DbSession,
):
    """Run only the ability-diagnosis stage and persist its evidence."""
    record = await db.scalar(
        select(HumanAssessment).where(
            HumanAssessment.id == session_id,
            HumanAssessment.user_id == user_id,
        )
    )
    if not record:
        raise HTTPException(status_code=404, detail="Planning session not found")
    if record.status == "diagnosed":
        raise HTTPException(status_code=409, detail="能力诊断已完成，请确认后开始规划")
    if record.status == "completed":
        raise HTTPException(status_code=409, detail="该测试已经完成规划")
    expected_question_ids = {item["id"] for item in record.questions}
    if set(payload.answers) != expected_question_ids:
        raise HTTPException(status_code=400, detail="请完成全部测试题后再提交诊断")

    session_state = dict(record.result or {})
    task_id = session_state.get("task_id")
    task = await db.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == user_id)
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    graph = TaskGraph.model_validate(session_state.get("task_graph"))
    agent_overrides = session_state.get("agent_overrides") or {}
    diagnosis = ability_diagnosis_agent.run(
        record.questions,
        payload.answers,
        graph,
        agent_overrides.get(ability_diagnosis_agent.name),
    )
    record.answers = payload.answers
    record.result = {**session_state, "diagnosis": diagnosis}
    record.status = "diagnosed"
    task.status = "diagnosed"

    profile = await db.scalar(select(HumanProfile).where(HumanProfile.user_id == user_id))
    if not profile:
        profile = HumanProfile(user_id=user_id)
        db.add(profile)
    profile.capability_vector = {
        **diagnosis["capability"],
        **diagnosis["planning_capability"],
    }
    profile.decision_history = {
        "source": "automatic_planning_assessment",
        "assessment_id": record.id,
        "task_id": task.id,
        "design_requirement": record.design_requirement,
        "overall": diagnosis["overall"],
        "confidence": diagnosis["confidence"],
        "weakest_dimensions": diagnosis["weakest_dimensions"],
    }
    user = await db.scalar(select(User).where(User.id == user_id))
    if user:
        user.adaptive_profile = profile.capability_vector
    await db.commit()
    return {
        "task_id": task.id,
        "diagnosis": diagnosis,
        "agent_trace": [
            {"agent": problem_analysis_agent.name, "status": "completed"},
            {"agent": test_generation_agent.name, "status": "completed"},
            {
                "agent": ability_diagnosis_agent.name,
                "status": "completed",
                "summary": f"综合能力 {diagnosis['overall']:.0%}",
            },
            {
                "agent": capability_planning_agent.name,
                "status": "waiting_for_confirmation",
                "summary": "等待用户确认后开始规划",
            },
        ],
    }


@router.post("/planning-sessions/{session_id}/plan")
async def create_workflow_from_diagnosis(
    session_id: str,
    payload: PlanningCreateWorkflowRequest,
    user_id: CurrentUserId,
    db: DbSession,
):
    """Create a workflow only after the user confirms a persisted diagnosis."""
    record = await db.scalar(
        select(HumanAssessment).where(
            HumanAssessment.id == session_id,
            HumanAssessment.user_id == user_id,
        )
    )
    if not record:
        raise HTTPException(status_code=404, detail="Planning session not found")
    if record.status == "completed":
        raise HTTPException(status_code=409, detail="该诊断已经生成工作流")
    if record.status != "diagnosed":
        raise HTTPException(status_code=409, detail="请先完成能力诊断，再开始规划")

    session_state = dict(record.result or {})
    diagnosis = session_state.get("diagnosis")
    if not diagnosis:
        raise HTTPException(status_code=409, detail="诊断结果不存在，请重新完成能力测试")
    task_id = session_state.get("task_id")
    task = await db.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == user_id)
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    graph = TaskGraph.model_validate(session_state.get("task_graph"))

    plan = await capability_planning_agent.run_async(
        graph,
        payload.capability_space,
        diagnosis,
        payload.weights,
        await model_config_for(db, user_id),
        (session_state.get("agent_overrides") or {}).get(capability_planning_agent.name),
    )
    workflow = Workflow(
        owner_id=user_id,
        task_id=task.id,
        name=f"{task.title} · 自适应工作流",
        definition=plan.model_dump(mode="json"),
        decision_trace={"items": plan.decision_trace},
        status="ready",
    )
    db.add(workflow)
    await db.flush()
    plan.id = workflow.id
    workflow.definition = plan.model_dump(mode="json")
    task.status = "planned"
    record.result = {**session_state, "workflow_id": workflow.id}
    record.status = "completed"
    await db.commit()
    return {
        "task_graph": graph,
        "diagnosis": diagnosis,
        "workflow": plan,
        "agent_trace": [
            {"agent": problem_analysis_agent.name, "status": "completed"},
            {"agent": test_generation_agent.name, "status": "completed"},
            {
                "agent": ability_diagnosis_agent.name,
                "status": "completed",
                "summary": f"综合能力 {diagnosis['overall']:.0%}",
            },
            {
                "agent": capability_planning_agent.name,
                "status": "completed",
                "summary": f"已生成 {len(plan.nodes)} 个协同节点",
            },
        ],
    }


@router.delete("/planning-sessions/{session_id}")
async def cancel_planning_session(
    session_id: str,
    user_id: CurrentUserId,
    db: DbSession,
):
    record = await db.scalar(
        select(HumanAssessment).where(
            HumanAssessment.id == session_id,
            HumanAssessment.user_id == user_id,
        )
    )
    if not record:
        raise HTTPException(status_code=404, detail="Planning session not found")
    if record.status == "completed":
        raise HTTPException(status_code=409, detail="已完成的规划不能作为临时测试取消")
    task_id = (record.result or {}).get("task_id")
    task = await db.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == user_id)
    )
    await db.delete(record)
    if task:
        await db.delete(task)
    await db.commit()
    return {"deleted": True}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, user_id: CurrentUserId, db: DbSession):
    task = await db.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == user_id)
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    graph = await db.scalar(select(TaskGraphRecord).where(TaskGraphRecord.task_id == task.id))
    workflow = await db.scalar(
        select(Workflow)
        .where(Workflow.task_id == task.id, Workflow.owner_id == user_id)
        .order_by(Workflow.updated_at.desc())
    )
    pending_session = None
    if not workflow:
        assessment_records = list(
            (
                await db.scalars(
                    select(HumanAssessment)
                    .where(
                        HumanAssessment.user_id == user_id,
                        HumanAssessment.status.in_(["awaiting_answers", "diagnosed"]),
                    )
                    .order_by(HumanAssessment.updated_at.desc())
                )
            ).all()
        )
        assessment_record = next(
            (
                item
                for item in assessment_records
                if (item.result or {}).get("task_id") == task.id
            ),
            None,
        )
        if assessment_record:
            diagnosis = (assessment_record.result or {}).get("diagnosis")
            diagnosed = assessment_record.status == "diagnosed"
            pending_session = {
                "session_id": assessment_record.id,
                "task_id": task.id,
                "status": assessment_record.status,
                "questions": assessment.public_questions(assessment_record.questions),
                "diagnosis": diagnosis,
                "agent_trace": [
                    {
                        "agent": problem_analysis_agent.name,
                        "status": "completed",
                        "summary": f"已解析为 {len(graph.graph.get('subtasks', [])) if graph else 0} 个子任务",
                    },
                    {
                        "agent": test_generation_agent.name,
                        "status": "completed",
                        "summary": f"已生成 {len(assessment_record.questions)} 道任务自适应测试题",
                        "mode": (assessment_record.result or {}).get("test_generation_mode"),
                    },
                    {
                        "agent": ability_diagnosis_agent.name,
                        "status": "completed" if diagnosed else "waiting_for_answers",
                        "summary": f"综合能力 {diagnosis['overall']:.0%}" if diagnosis else None,
                    },
                    {
                        "agent": capability_planning_agent.name,
                        "status": "waiting_for_confirmation" if diagnosed else "waiting_for_diagnosis",
                        "summary": "等待用户确认后开始规划" if diagnosed else None,
                    },
                ],
            }
    return {
        "task": {
            "id": task.id,
            "title": task.title,
            "prompt": task.prompt,
            "status": task.status,
            "task_type": task.task_type,
            "complexity": task.complexity,
            "constraints": task.constraints,
            "updated_at": task.updated_at.isoformat(),
        },
        "task_graph": graph.graph if graph else None,
        "workflow": public_workflow_definition(workflow.definition) if workflow else None,
        "planning_session": pending_session,
    }


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, user_id: CurrentUserId, db: DbSession):
    task = await db.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == user_id)
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    assessment_records = list(
        (
            await db.scalars(
                select(HumanAssessment).where(HumanAssessment.user_id == user_id)
            )
        ).all()
    )
    for assessment_record in assessment_records:
        if (assessment_record.result or {}).get("task_id") == task.id:
            await db.delete(assessment_record)
    await db.delete(task)
    await db.commit()
    return {"deleted": True, "task_id": task_id}


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
