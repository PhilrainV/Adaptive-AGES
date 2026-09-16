from enum import StrEnum
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class SubjectType(StrEnum):
    LLM = "llm"
    ML = "ml"
    HUMAN = "human"
    TOOL = "tool"


class ExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"


class EdgeType(StrEnum):
    DEFAULT = "default"
    CONDITIONAL = "conditional"
    LOOP = "loop"


class UserCreate(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AgentSkill(BaseModel):
    """A user-supplied instruction block injected into one planning agent."""

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    instructions: str = Field(default="", max_length=4000)
    enabled: bool = True


class AgentRuntimeConfig(BaseModel):
    """Per-session extension point; safe defaults live beside each agent."""

    system_prompt: str | None = Field(default=None, max_length=12000)
    skills: list[AgentSkill] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentCreate(BaseModel):
    name: str
    agent_type: SubjectType
    description: str | None = None
    endpoint: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, float] = Field(default_factory=dict)


class CapabilityRequirement(BaseModel):
    reasoning: float = 0
    prediction: float = 0
    generation: float = 0
    interpretation: float = 0
    domain_knowledge: float = 0
    human_judgement: float = 0
    data_processing: float = 0


class SubjectSuitability(BaseModel):
    subject_type: SubjectType
    suitability: float = Field(ge=0, le=1)
    role: str = "executor"
    rationale: str = ""


class IterationPolicy(BaseModel):
    enabled: bool = False
    feedback_target_subtask_id: str | None = None
    condition: str = "needs_revision == true"
    max_iterations: int = Field(default=1, ge=1, le=10)


class Subtask(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    requirement: CapabilityRequirement
    input_contract: list[str] = Field(default_factory=list)
    output_contract: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    runtime_hints: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    risk: float = 0
    subject_suitability: list[SubjectSuitability] = Field(default_factory=list)
    preferred_subject_types: list[SubjectType] = Field(default_factory=list)
    unsuitable_subject_types: list[SubjectType] = Field(default_factory=list)
    assignment_rationale: str = ""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    entry_condition: str | None = None
    iteration_policy: IterationPolicy = Field(default_factory=IterationPolicy)


class TaskGraph(BaseModel):
    task_id: str
    goal: str
    complexity: float = Field(ge=0, le=1)
    subtasks: list[Subtask]
    planning_mode: str = "rule"
    assignment_summary: dict[str, list[str]] = Field(default_factory=dict)
    analysis_trace: list[dict[str, Any]] = Field(default_factory=list)


class TaskUnderstandRequest(BaseModel):
    prompt: str = Field(min_length=8)
    constraints: dict[str, Any] = Field(default_factory=dict)


class CapabilitySubject(BaseModel):
    id: str
    name: str
    subject_type: SubjectType
    capability: dict[str, float]
    reliability: float = .8
    cost: float = .2
    latency: float = .2
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchResult(BaseModel):
    subject_id: str
    subject_name: str
    subject_type: SubjectType
    score: float
    similarity: float
    explanation: str
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


class PlanRequest(BaseModel):
    task_graph: TaskGraph
    capability_space: list[CapabilitySubject]
    user_profile: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fit": .42,
            "reliability": .15,
            "comfort": .18,
            "complementarity": .12,
            "cost": .08,
            "latency": .05,
        }
    )


class WorkflowNode(BaseModel):
    id: str
    subtask_id: str
    subject_id: str
    subject_type: SubjectType
    label: str
    match_score: float
    config: dict[str, Any] = Field(default_factory=dict)
    explainability: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    source: str
    target: str
    condition: str | None = None
    edge_type: EdgeType = EdgeType.DEFAULT
    max_iterations: int = Field(default=1, ge=1, le=10)


class WorkflowPlan(BaseModel):
    id: str
    task_id: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]
    decision_trace: list[dict[str, Any]]
    estimated_cost: float
    requires_human: bool


class FeedbackCreate(BaseModel):
    node_id: str | None = None
    feedback_type: str
    score: float | None = Field(default=None, ge=0, le=1)
    correction: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class WorkflowUpdate(BaseModel):
    name: str | None = None
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]


class WorkflowRunRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


class ModelSettingsUpdate(BaseModel):
    provider: str = "openai-compatible"
    model: str = "gpt-4.1-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=.2, ge=0, le=2)


class NodeModelSettingsUpdate(ModelSettingsUpdate):
    use_default: bool = True
    modality: str = "text"


class PlanningStartRequest(TaskUnderstandRequest):
    assessment_enabled: bool = True
    capability_space: list[CapabilitySubject] = Field(default_factory=list)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fit": .42,
            "reliability": .15,
            "comfort": .18,
            "complementarity": .12,
            "cost": .08,
            "latency": .05,
        }
    )
    agent_overrides: dict[str, AgentRuntimeConfig] = Field(default_factory=dict)


class PlanningDiagnoseRequest(BaseModel):
    answers: dict[str, int]


class PlanningCreateWorkflowRequest(BaseModel):
    capability_space: list[CapabilitySubject]
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fit": .42,
            "reliability": .15,
            "comfort": .18,
            "complementarity": .12,
            "cost": .08,
            "latency": .05,
        }
    )


class HumanAssessmentGenerateRequest(BaseModel):
    design_requirement: str = Field(min_length=8)


class HumanAssessmentSubmitRequest(BaseModel):
    answers: dict[str, int]
