from uuid import uuid4
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from app.api.deps import CurrentUserId, DbSession
from app.core.security import create_access_token, hash_password, verify_password
from app.models.entities import Agent, AgentCapability, Feedback, Task, TaskGraph as TaskGraphRecord, User, Workflow, WorkflowExecution
from app.schemas.domain import AgentCreate, FeedbackCreate, LoginRequest, PlanRequest, TaskUnderstandRequest, UserCreate, WorkflowPlan
from app.services.adaptive_planner import AdaptivePlanner
from app.services.task_understanding import TaskUnderstandingEngine
from app.workflow.langgraph_engine import LangGraphExecutionEngine


router = APIRouter()
understanding = TaskUnderstandingEngine()
planner = AdaptivePlanner()
engine = LangGraphExecutionEngine()


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: DbSession):
    if await db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=payload.email, display_name=payload.display_name, password_hash=hash_password(payload.password), adaptive_profile={"coding_skill": .5, "ai_skill": .5, "domain_skill": .5})
    db.add(user); await db.commit(); await db.refresh(user)
    return {"id": user.id, "email": user.email, "display_name": user.display_name, "access_token": create_access_token(user.id), "token_type": "bearer"}


@router.post("/auth/login")
async def login(payload: LoginRequest, db: DbSession):
    user = await db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": create_access_token(user.id), "token_type": "bearer"}


@router.get("/agents")
async def list_agents(user_id: CurrentUserId, db: DbSession):
    agents = (await db.scalars(select(Agent).where(Agent.owner_id == user_id))).all()
    return [{"id": a.id, "name": a.name, "agent_type": a.agent_type, "description": a.description, "config": a.config, "is_active": a.is_active} for a in agents]


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreate, user_id: CurrentUserId, db: DbSession):
    agent = Agent(owner_id=user_id, name=payload.name, agent_type=payload.agent_type.value, description=payload.description, endpoint=payload.endpoint, config=payload.config)
    for dimension, score in payload.capabilities.items(): agent.capabilities.append(AgentCapability(dimension=dimension, score=score, confidence=.6))
    db.add(agent); await db.commit(); await db.refresh(agent)
    return {"id": agent.id, "name": agent.name, "agent_type": agent.agent_type}


@router.post("/tasks/understand")
async def understand_task(payload: TaskUnderstandRequest, user_id: CurrentUserId, db: DbSession):
    task = Task(owner_id=user_id, title=payload.prompt[:120], prompt=payload.prompt, constraints=payload.constraints)
    db.add(task); await db.flush()
    graph = understanding.understand(payload.prompt)
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
    record = Workflow(owner_id=user_id, task_id=task.id, name=f"{task.title} · 自适应工作流", definition=plan.model_dump(mode="json"), decision_trace={"items": plan.decision_trace}, status="ready")
    db.add(record); await db.flush()
    plan.id = record.id
    record.definition = plan.model_dump(mode="json")
    await db.commit()
    return plan


@router.post("/workflows/execute")
async def execute_workflow(plan: WorkflowPlan, user_id: CurrentUserId, db: DbSession):
    workflow = await db.scalar(select(Workflow).where(Workflow.id == plan.id, Workflow.owner_id == user_id))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    execution = WorkflowExecution(workflow_id=workflow.id, status="running", input={})
    db.add(execution); await db.flush()
    result = await engine.execute(plan, execution.id, {})
    execution.status = result.get("status", "completed")
    execution.state = dict(result)
    execution.output = result.get("node_outputs", {})
    await db.commit()
    return {"execution_id": execution.id, **result}


@router.post("/executions/{execution_id}/human-feedback")
async def submit_human_feedback(execution_id: str, payload: FeedbackCreate, user_id: CurrentUserId, db: DbSession):
    execution = await db.scalar(select(WorkflowExecution).join(Workflow).where(WorkflowExecution.id == execution_id, Workflow.owner_id == user_id))
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    record = Feedback(execution_id=execution.id, user_id=user_id, node_id=payload.node_id, feedback_type=payload.feedback_type, score=payload.score, correction=payload.correction, rationale=payload.rationale)
    db.add(record)
    execution.status = "resumable"
    await db.commit()
    return {"execution_id": execution_id, "status": "resumable", "feedback_id": record.id}
