# Adaptive planning research agents

Automatic planning is intentionally split into four persisted stages. Each agent
owns one research question and exposes local constants plus a per-session override.

## 1. ProblemAnalysisAgent

File: `backend/app/agents/problem_analysis_agent.py`

- Owns its complete system prompt instead of forwarding directly to a service.
- Produces capability requirements, LLM/ML/Human/Tool suitability, preferred and
  unsuitable subject types, risk, dependencies and control-flow proposals.
- Persists `assignment_summary` and `analysis_trace` in the task graph.
- Uses `TaskUnderstandingEngine` only as an auditable fallback.

Extension points: `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_SKILLS`,
`_default_suitability`, `_clean_subtasks`.

## 2. TestGenerationAgent

File: `backend/app/agents/test_generation_agent.py`

- Generates task-grounded scenario items rather than generic AI questions.
- Adds `knowledge_components` to every item. Together these rows form the Q matrix.
- Adds initial DINA item parameters (`guess` and `slip`) and descriptive item
  parameters (`difficulty` and `discrimination`).
- Rejects a generated test unless every attribute has two observations and at
  least one single-attribute anchor item.

Extension points: `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_SKILLS`,
`ASSESSMENT_BLUEPRINT`, `validate_blueprint`.

## 3. AbilityDiagnosisAgent

File: `backend/app/agents/ability_diagnosis_agent.py`

- Uses an exact Bayesian DINA estimator over 32 mastery classes for five attributes.
- Returns posterior mastery probabilities, entropy, Q-matrix coverage, item
  residuals, confidence and the most likely mastery patterns.
- Computes task-weighted overall ability and maps diagnosed attributes into the
  shared planning capability space.
- Accepts another estimator through the `CognitiveDiagnosisEstimator` protocol,
  allowing future G-DINA, NIDA, IRT or neural cognitive diagnosis experiments.

Extension points: `BayesianDINA`, `DIMENSIONS`, `PLANNING_CAPABILITY_MAP`, estimator
injection, and parameters such as `mastery_prior`, `default_guess`, `default_slip`,
`mastered_threshold` and `developing_threshold`.

## 4. CapabilityPlanningAgent

File: `backend/app/agents/capability_planning_agent.py`

- Runs a five-stage hybrid pipeline: LLM candidate generation, multi-objective
  optimization, LLM plus deterministic Critic review, automatic constraint repair,
  and final utility-based selection.
- The LLM receives the persisted task graph, cognitive diagnosis, available
  subjects and enabled custom Skills. It proposes multiple genuinely different
  workflow strategies rather than directly producing the final graph.
- Calibrates Human subjects with the current task's cognitive diagnosis and uses
  beam-search baselines to prevent unconstrained model output from becoming a plan.
- The objective includes absolute requirement coverage, problem-analysis subject
  prior, reliability, user comfort, machine complementarity, cost, latency, risk
  governance, subject load and type diversity.
- Adds structured machine scaffolding to human nodes when the user's diagnosed
  gap is material.
- The Critic checks coverage, invalid or unsuitable assignments, human overload,
  parallel opportunities, missing branch conditions, failure/review paths and
  bounded loops. Deterministic repair is applied before candidates are ranked.
- Persists every candidate, objective score, Critic issue, repair and final
  selection in `decision_trace` so the decision can be audited and reproduced.
- When no model API is configured or a provider call fails, the same pipeline falls
  back to three deterministic strategies instead of blocking workflow creation.

Extension points: `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_CRITIC_PROMPT`, `DEFAULT_SKILLS`,
`DEFAULT_WEIGHTS`, `DEFAULT_POLICY`, `_generate_llm_candidates`,
`_criticise_with_llm`, `_score_candidate`, `_repair_candidate`, `_optimise`,
`_synthesise`, and `_node_config`.

## Per-session prompt, skill and parameter overrides

`POST /api/v1/planning-sessions/start` accepts optional `agent_overrides`. Overrides
are stored with the planning session and are reused by diagnosis and planning.

```json
{
  "prompt": "分析学生数据，预测风险，生成建议并由教师复核",
  "constraints": {},
  "agent_overrides": {
    "problem_analysis_agent": {
      "system_prompt": "在默认任务分析要求基础上，重点识别教育责任边界。",
      "skills": [
        {
          "name": "education_safety",
          "description": "教育高风险识别",
          "instructions": "涉及学生权益时必须保留教师最终判断。",
          "enabled": true
        }
      ]
    },
    "ability_diagnosis_agent": {
      "parameters": {
        "mastery_prior": 0.5,
        "default_guess": 0.22,
        "default_slip": 0.12
      }
    },
    "capability_planning_agent": {
      "system_prompt": "优先生成可解释且适合教师实际操作的多主体候选工作流。",
      "skills": [
        {
          "name": "education_accountability",
          "description": "教育决策责任边界",
          "instructions": "高风险学生决策必须由教师确认，并保留机器建议证据。",
          "enabled": true
        }
      ],
      "parameters": {
        "candidate_count": 4,
        "beam_width": 96,
        "comfort_target_gap": 0.05,
        "human_overload_limit": 0.2,
        "human_review_risk": 0.75
      }
    }
  }
}
```

Only the model API key remains in encrypted model settings. Agent prompts, skills
and algorithm parameters contain no secrets and travel with the provisional
planning session for reproducibility.

## Control-flow contract

- `execution_mode=parallel`: sibling tasks sharing dependencies may fan out.
- `entry_condition`: creates a conditional edge using safe comparisons such as
  `confidence < 0.7` or `approved == true`.
- `iteration_policy`: creates a bounded loop, commonly
  `needs_revision == true`, with `max_iterations` from 1 to 10.
- The server executor and exported runner parse only simple field comparisons.
  They do not use Python `eval`.

Do not add a branch or loop merely to make a graph look complex. Control flow must
correspond to a real uncertainty threshold, missing-data path, review decision or
revision mechanism.
