import runpy
from collections import Counter

import pytest

from app.agents import (
    AbilityDiagnosisAgent,
    CapabilityPlanningAgent,
    ProblemAnalysisAgent,
    TestGenerationAgent,
)
from app.agents.capability_planning_agent import (
    CandidateCritiqueSet,
    CandidateProposal,
    CandidateProposalSet,
    ProposedAssignment,
)
from app.schemas.domain import (
    AgentRuntimeConfig,
    AgentSkill,
    CapabilityRequirement,
    CapabilitySubject,
    EdgeType,
    ExecutionMode,
    IterationPolicy,
    SubjectSuitability,
    SubjectType,
    Subtask,
    TaskGraph,
    WorkflowEdge,
    WorkflowNode,
    WorkflowPlan,
)
from app.services.human_assessment import HumanCapabilityAssessmentService
from app.workflow.langgraph_engine import LangGraphExecutionEngine


@pytest.mark.asyncio
async def test_problem_agent_persists_subject_analysis_control_flow_and_custom_skill():
    agent = ProblemAnalysisAgent()
    graph = await agent.run(
        "分析学生数据，预测风险，生成建议并由教师复核",
        agent_config=AgentRuntimeConfig(
            skills=[AgentSkill(name="education_safety", instructions="保留教师最终责任")]
        ),
    )

    assert all(item.subject_suitability for item in graph.subtasks)
    assert set(graph.assignment_summary) == {"llm", "ml", "human", "tool"}
    assert "education_safety" in graph.analysis_trace[0]["skills"]
    review = next(item for item in graph.subtasks if item.task_type == "human_review")
    assert review.execution_mode == ExecutionMode.ITERATIVE
    assert review.iteration_policy.enabled is True


@pytest.mark.asyncio
async def test_problem_agent_repairs_under_decomposition_for_multi_stage_learning_loop(tmp_path):
    prompt = (
        "帮我制定学习提升方案。先分析学生当前知识水平，如果存在薄弱知识点，"
        "针对每个薄弱点生成练习；学生完成练习后检查学习效果，未达到目标则调整并重新生成，"
        "直到达标或达到最大循环次数，最终生成总结报告。"
    )
    graph = await ProblemAnalysisAgent().run(prompt)

    assert len(graph.subtasks) >= 10
    assert {item.task_type for item in graph.subtasks} >= {
        "prediction", "generation", "human_action", "evaluation",
    }
    assert any(item.entry_condition for item in graph.subtasks)
    assert any(item.iteration_policy.enabled for item in graph.subtasks)
    subjects = [
        CapabilitySubject(id="llm", name="LLM", subject_type=SubjectType.LLM, capability={"reasoning": .96, "generation": .96, "interpretation": .92, "domain_knowledge": .72}),
        CapabilitySubject(id="ml", name="ML", subject_type=SubjectType.ML, capability={"prediction": .97, "data_processing": .82, "interpretation": .76}),
        CapabilitySubject(id="human", name="Human", subject_type=SubjectType.HUMAN, capability={"human_judgement": .96, "domain_knowledge": .93, "interpretation": .9}),
        CapabilitySubject(id="tool", name="Tool", subject_type=SubjectType.TOOL, capability={"data_processing": .98, "prediction": .24, "interpretation": .34}),
    ]
    plan = CapabilityPlanningAgent().run(
        graph,
        subjects,
        {"method": "assessment_disabled", "planning_capability": {}, "weakest_dimensions": []},
    )
    selected_types = {node.subject_type for node in plan.nodes}
    assert {SubjectType.LLM, SubjectType.ML, SubjectType.HUMAN} <= selected_types
    assert any(edge.edge_type == EdgeType.LOOP for edge in plan.edges)
    type_counts = Counter(node.subject_type for node in plan.nodes)
    assert type_counts[SubjectType.LLM] >= 3
    assert type_counts[SubjectType.ML] >= 3
    assert type_counts[SubjectType.TOOL] >= 2

    nodes = {node.subtask_id: node for node in plan.nodes}
    diagnosis_node = nodes["estimate-mastery"]
    module_path = tmp_path / "mastery_node.py"
    module_path.write_text(diagnosis_node.config["code"], encoding="utf-8")
    namespace = runpy.run_path(module_path)
    result = namespace["run"](
        {
            "responses": [
                {"knowledge_point": "fractions", "correct": True},
                {"knowledge_point": "fractions", "correct": False},
                {"knowledge_point": "algebra", "score": .9},
            ]
        },
        {},
    )
    assert result["mastery_by_knowledge_point"]
    assert diagnosis_node.config["algorithm"] == "beta_binomial_mastery"

    practice_node = nodes["generate-practice"]
    assert "薄弱点练习生成" in practice_node.config["system_prompt"]
    assert practice_node.config["skills"]
    assert "exercises[]" in practice_node.config["output_contract"]

    tool_node = nodes["validate-practice"]
    assert tool_node.config["connector"] == "builtin"
    assert tool_node.config["operation"] == "validate_exercise_set"

    human_node = nodes["human-practice"]
    assert human_node.config["response_schema"]["required_fields"]


@pytest.mark.asyncio
async def test_emotion_aware_learning_request_creates_two_exclusive_branches_and_one_loop():
    prompt = (
        "帮我制定一个学习提升方案。首先分析学生当前知识水平和情绪，如果情绪不好走情绪安抚路线，"
        "先把情绪照顾好。如果情绪没有问题，存在薄弱知识点，则针对每个薄弱点生成练习任务；"
        "完成每个练习后检查学习效果，如果仍未达到目标，则继续调整任务并重新生成练习，直到达到目标水平。"
        "所以任务要做两个分支，然后其中一个分支有循环。"
    )
    graph = await ProblemAnalysisAgent().run(prompt)
    tasks = {item.id: item for item in graph.subtasks}

    assert tasks["emotion-support"].entry_condition == "emotion_needs_support == true"
    assert tasks["rank-weak-points"].entry_condition == "emotion_needs_support == false"
    assert tasks["emotion-support"].dependencies == ["emotion-router"]
    assert tasks["rank-weak-points"].dependencies == ["emotion-router"]
    assert tasks["analyse-feedback"].iteration_policy.feedback_target_subtask_id == "design-practice"

    subjects = [
        CapabilitySubject(id="llm", name="LLM", subject_type=SubjectType.LLM, capability={"reasoning": .96, "generation": .96, "interpretation": .94, "domain_knowledge": .76, "human_judgement": .72}),
        CapabilitySubject(id="ml", name="ML", subject_type=SubjectType.ML, capability={"prediction": .97, "data_processing": .84, "interpretation": .76}),
        CapabilitySubject(id="human", name="Human", subject_type=SubjectType.HUMAN, capability={"human_judgement": .96, "domain_knowledge": .9, "interpretation": .9}),
        CapabilitySubject(id="tool", name="Tool", subject_type=SubjectType.TOOL, capability={"data_processing": .98, "prediction": .24, "interpretation": .72}),
    ]
    plan = CapabilityPlanningAgent().run(
        graph,
        subjects,
        {"method": "assessment_disabled", "planning_capability": {}, "weakest_dimensions": []},
    )
    router_id = next(node.id for node in plan.nodes if node.subtask_id == "emotion-router")
    outgoing = [edge for edge in plan.edges if edge.source == router_id]

    assert len(outgoing) == 2
    assert {edge.condition for edge in outgoing} == {
        "emotion_needs_support == true",
        "emotion_needs_support == false",
    }
    assert all(edge.edge_type == EdgeType.CONDITIONAL for edge in outgoing)
    loops = [edge for edge in plan.edges if edge.edge_type == EdgeType.LOOP]
    assert len(loops) == 1
    assert loops[0].source == "node-analyse-feedback"
    assert loops[0].target == "node-design-practice"


@pytest.mark.asyncio
async def test_test_generation_produces_an_identifiable_q_matrix():
    graph = await ProblemAnalysisAgent().run("分析学生数据，预测风险，生成建议并由教师复核")
    questions, mode = await TestGenerationAgent().run(graph)

    TestGenerationAgent.validate_blueprint(questions)
    assert len(questions) == 10
    assert mode == "dina_rule"
    assert all(item["knowledge_components"] for item in questions)
    assert all("guess" in item and "slip" in item for item in questions)


def test_bayesian_dina_estimates_attribute_mastery_instead_of_raw_accuracy():
    questions = HumanCapabilityAssessmentService().generate("构建教育风险预测与教师复核系统")
    correct = {item["id"]: item["correct_index"] for item in questions}
    weak_programming = dict(correct)
    for item in questions:
        if item["dimension"] == "programming":
            weak_programming[item["id"]] = (item["correct_index"] + 1) % 4

    strong = AbilityDiagnosisAgent().run(questions, correct)
    diagnosed = AbilityDiagnosisAgent().run(questions, weak_programming)

    assert diagnosed["method"] == "bayesian_dina"
    assert diagnosed["capability"]["programming"] < strong["capability"]["programming"]
    assert diagnosed["q_matrix_coverage"]["programming"] == 2
    assert len(diagnosed["top_mastery_patterns"]) == 5
    assert 0 <= diagnosed["confidence"] <= 1


def test_global_planner_balances_subjects_and_preserves_non_linear_flow():
    fit = lambda kind, score: SubjectSuitability(subject_type=kind, suitability=score)
    graph = TaskGraph(
        task_id="task-1",
        goal="预测、解释并复核",
        complexity=.8,
        subtasks=[
            Subtask(
                id="prepare", name="准备", description="清洗数据", task_type="data_processing",
                requirement=CapabilityRequirement(data_processing=.95),
                subject_suitability=[fit(SubjectType.TOOL, .95), fit(SubjectType.ML, .8)],
                preferred_subject_types=[SubjectType.TOOL],
            ),
            Subtask(
                id="predict", name="预测", description="预测风险", task_type="prediction",
                requirement=CapabilityRequirement(prediction=.95, data_processing=.7), dependencies=["prepare"],
                subject_suitability=[fit(SubjectType.ML, .98), fit(SubjectType.LLM, .4)],
                preferred_subject_types=[SubjectType.ML], execution_mode=ExecutionMode.PARALLEL,
            ),
            Subtask(
                id="explain", name="解释", description="生成解释", task_type="generation",
                requirement=CapabilityRequirement(reasoning=.85, generation=.95), dependencies=["prepare"],
                subject_suitability=[fit(SubjectType.LLM, .98), fit(SubjectType.HUMAN, .65)],
                preferred_subject_types=[SubjectType.LLM], execution_mode=ExecutionMode.PARALLEL,
            ),
            Subtask(
                id="review", name="复核", description="教师复核", task_type="human_review", risk=.9,
                requirement=CapabilityRequirement(human_judgement=.95, domain_knowledge=.9), dependencies=["predict", "explain"],
                subject_suitability=[fit(SubjectType.HUMAN, .99), fit(SubjectType.LLM, .45)],
                preferred_subject_types=[SubjectType.HUMAN], execution_mode=ExecutionMode.ITERATIVE,
                iteration_policy=IterationPolicy(enabled=True, feedback_target_subtask_id="explain", max_iterations=2),
            ),
        ],
    )
    subjects = [
        CapabilitySubject(id="llm", name="LLM", subject_type=SubjectType.LLM, capability={"reasoning":.96,"generation":.98,"interpretation":.9}),
        CapabilitySubject(id="ml", name="ML", subject_type=SubjectType.ML, capability={"prediction":.98,"data_processing":.9}),
        CapabilitySubject(id="human", name="Human", subject_type=SubjectType.HUMAN, capability={}, reliability=.9, cost=.7, latency=.8),
        CapabilitySubject(id="tool", name="Tool", subject_type=SubjectType.TOOL, capability={"data_processing":.99}, cost=.03, latency=.03),
    ]
    diagnosis = {
        "method":"bayesian_dina", "overall":.62, "confidence":.8,
        "weakest_dimensions":["programming"],
        "planning_capability":{"reasoning":.65,"generation":.62,"prediction":.35,"data_processing":.3,"human_judgement":.86,"domain_knowledge":.82,"interpretation":.7},
    }

    plan = CapabilityPlanningAgent().run(graph, subjects, diagnosis)
    selected = {item.subtask_id: item.subject_type for item in plan.nodes}

    assert selected["predict"] == SubjectType.ML
    assert selected["explain"] == SubjectType.LLM
    assert selected["review"] == SubjectType.HUMAN
    assert any(edge.edge_type == EdgeType.LOOP for edge in plan.edges)
    assert plan.decision_trace[0]["agent"] == "capability_planning_agent"
    assert plan.decision_trace[1]["workflow_topology"]["parallel_fan_outs"] >= 1


@pytest.mark.asyncio
async def test_hybrid_planner_runs_generation_critic_repair_and_selection(monkeypatch):
    graph = TaskGraph(
        task_id="hybrid-task",
        goal="生成高风险建议并确认",
        complexity=.7,
        subtasks=[
            Subtask(
                id="advise",
                name="生成建议",
                description="生成需要专业责任确认的建议",
                task_type="generation",
                risk=.9,
                requirement=CapabilityRequirement(reasoning=.8, generation=.9, human_judgement=.7),
                subject_suitability=[
                    SubjectSuitability(subject_type=SubjectType.LLM, suitability=.92),
                    SubjectSuitability(subject_type=SubjectType.HUMAN, suitability=.8),
                ],
                preferred_subject_types=[SubjectType.LLM],
            )
        ],
    )
    subjects = [
        CapabilitySubject(
            id="llm", name="LLM", subject_type=SubjectType.LLM,
            capability={"reasoning": .95, "generation": .98}, reliability=.9,
        ),
        CapabilitySubject(
            id="human", name="Human", subject_type=SubjectType.HUMAN,
            capability={}, reliability=.9, cost=.7, latency=.8,
        ),
    ]
    diagnosis = {
        "method": "bayesian_dina",
        "overall": .58,
        "confidence": .81,
        "weakest_dimensions": ["programming"],
        "planning_capability": {"reasoning": .55, "generation": .5, "human_judgement": .86},
    }
    proposals = CandidateProposalSet(candidates=[
        CandidateProposal(
            name="机器生成后确认",
            strategy="quality_first",
            assignments=[ProposedAssignment(subtask_id="advise", subject_id="llm")],
        ),
        CandidateProposal(
            name="人类直接处理",
            strategy="human_first",
            assignments=[ProposedAssignment(subtask_id="advise", subject_id="human")],
        ),
    ])

    async def fake_generate(*_args, **_kwargs):
        return proposals

    async def fake_critic(*_args, **_kwargs):
        return CandidateCritiqueSet(critiques=[])

    agent = CapabilityPlanningAgent()
    monkeypatch.setattr(agent, "_generate_llm_candidates", fake_generate)
    monkeypatch.setattr(agent, "_criticise_with_llm", fake_critic)
    plan = await agent.run_async(
        graph,
        subjects,
        diagnosis,
        model_config={"api_key": "test", "model": "mock-model"},
        agent_config=AgentRuntimeConfig(
            skills=[AgentSkill(name="education_accountability", instructions="专业建议保留人类责任")]
        ),
    )

    trace = plan.decision_trace[0]
    assert [item["stage"] for item in trace["pipeline"]] == [
        "candidate_generation",
        "multi_objective_optimisation",
        "critic",
        "constraint_repair",
        "final_selection",
    ]
    assert trace["pipeline"][0]["mode"] == "llm"
    assert trace["pipeline"][2]["mode"] == "llm_plus_constraints"
    assert "education_accountability" in trace["skills"]
    assert any(item["source"] == "llm" for item in trace["candidate_decisions"])
    assert any(node.subtask_id == "advise-human-check" for node in plan.nodes)


def test_direct_planning_does_not_use_a_human_profile_or_comfort_objectives():
    graph = TaskGraph(
        task_id="direct-task",
        goal="直接生成摘要",
        complexity=.3,
        subtasks=[
            Subtask(
                id="summarise",
                name="生成摘要",
                description="总结给定材料",
                task_type="generation",
                requirement=CapabilityRequirement(reasoning=.6, generation=.8),
                preferred_subject_types=[SubjectType.LLM],
            )
        ],
    )
    plan = CapabilityPlanningAgent().run(
        graph,
        [
            CapabilitySubject(
                id="llm", name="LLM", subject_type=SubjectType.LLM,
                capability={"reasoning": .9, "generation": .95},
            )
        ],
        {
            "method": "assessment_disabled",
            "personalization_enabled": False,
            "capability": {},
            "planning_capability": {},
            "weakest_dimensions": [],
            "confidence": 0,
        },
    )

    trace = plan.decision_trace[0]
    assert trace["human_assessment"]["personalization_enabled"] is False
    assert trace["objective_weights"]["comfort"] == 0
    assert trace["objective_weights"]["complementarity"] == 0
    assert all(item["strategy"] != "comfort_first" for item in trace["candidate_decisions"])


def test_condition_evaluator_is_bounded_and_does_not_use_eval():
    engine = LangGraphExecutionEngine()
    assert engine._condition_matches("confidence < 0.7", {"confidence": .4}, {}) is True
    assert engine._condition_matches("needs_revision == true", {"needs_revision": True}, {}) is True
    assert engine._condition_matches("__import__('os').system('echo unsafe')", {}, {}) is False


@pytest.mark.asyncio
async def test_langgraph_executes_a_bounded_feedback_loop():
    nodes = [
        WorkflowNode(
            id=name,
            subtask_id=name,
            subject_id="tool",
            subject_type=SubjectType.TOOL,
            label=name,
            match_score=1,
            config={"connector": "passthrough"},
        )
        for name in ["prepare", "generate", "review"]
    ]
    plan = WorkflowPlan(
        id="workflow-loop",
        task_id="task-loop",
        nodes=nodes,
        edges=[
            WorkflowEdge(source="prepare", target="generate"),
            WorkflowEdge(source="generate", target="review"),
            WorkflowEdge(
                source="review",
                target="generate",
                condition="payload.revise == true",
                edge_type=EdgeType.LOOP,
                max_iterations=2,
            ),
        ],
        decision_trace=[],
        estimated_cost=0,
        requires_human=False,
    )

    result = await LangGraphExecutionEngine().execute(plan, "execution-loop", {"revise": True})

    assert result["status"] == "completed"
    assert result["iteration_counts"]["review->generate"] == 2
