# Adaptive-AGES API

Base URL: `/api/v1`. Protected routes use `Authorization: Bearer <token>`.

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/auth/register` | Create a user and initialize the adaptive profile |
| POST | `/auth/login` | Issue an access token |
| GET | `/agents` | List the current user's execution subjects |
| POST | `/agents` | Register an LLM, ML, human, or tool subject and its capability vector |
| GET | `/dashboard` | Return real task, workflow, execution, and aggregate workspace data |
| POST | `/planning-sessions/start` | Analyse a problem, then either start assessment or directly create a workflow |
| POST | `/planning-sessions/{id}/diagnose` | Score all answers, persist the diagnosis, and stop before planning |
| POST | `/planning-sessions/{id}/plan` | Create a workflow only after explicit user confirmation |
| DELETE | `/planning-sessions/{id}` | Cancel an unfinished test and remove its provisional task |
| POST | `/tasks/understand` | Convert a natural-language goal to a task graph |
| GET | `/tasks/{id}` | Load a task, its graph, workflow, or pending assessment |
| DELETE | `/tasks/{id}` | Delete an owned task and its related workflow/execution records |
| POST | `/workflows/plan` | Match subjects and generate an explainable workflow |
| PUT | `/workflows/{id}` | Persist edited nodes, edges, prompts, code and review criteria |
| POST | `/workflows/{id}/execute` | Execute the saved graph with a task input |
| GET | `/workflows/{id}/export` | Download an independent executable workflow ZIP |
| POST | `/workflows/execute` | Compile and execute a workflow through LangGraph |
| POST | `/executions/{id}/human-feedback` | Submit a human decision or correction and make a paused run resumable |
| GET/PUT | `/settings/model` | Read or save encrypted OpenAI-compatible model configuration |
| POST | `/settings/model/test` | Validate the stored API configuration |
| GET | `/human-assessments/profile` | Read evidence-backed human capability values |

## Core exchange

1. `POST /planning-sessions/start` accepts two explicit modes.

Assessment-first mode:

```json
{
  "prompt": "分析学生学习数据，预测学业风险，生成个性化教学建议并由教师复核",
  "constraints": {"max_latency_seconds": 120, "explainable": true},
  "assessment_enabled": true,
  "capability_space": []
}
```

The problem-analysis agent creates the task graph and the test-generation agent
creates the task-grounded assessment. After all answers are submitted,
`POST /planning-sessions/{id}/diagnose` runs Bayesian DINA and persists the
diagnosis. The workflow is created only after explicit confirmation through
`POST /planning-sessions/{id}/plan`.

Direct mode:

```json
{
  "prompt": "分析学生学习数据并生成风险干预工作流",
  "constraints": {},
  "assessment_enabled": false,
  "capability_space": [
    {
      "id": "llm-general",
      "name": "General LLM",
      "subject_type": "llm",
      "capability": {"reasoning": 0.9, "generation": 0.9},
      "reliability": 0.85,
      "cost": 0.2,
      "latency": 0.2
    }
  ]
}
```

Direct mode skips both TestGenerationAgent and AbilityDiagnosisAgent. It does not
read or update HumanProfile. Human comfort and machine-complementarity objectives
are disabled; the planner uses only the task graph and the declared general
capability, reliability, cost and latency of available subjects. The same request
returns the completed workflow.

2. In either mode, CapabilityPlanningAgent runs candidate generation,
multi-objective optimization, Critic review, constraint repair and final selection.
Optional `agent_overrides` supply custom prompts, Skills and algorithm parameters.

3. The client may edit and persist the plan with `PUT /workflows/{id}`.
`POST /workflows/{id}/execute` compiles the saved graph. A human node changes the
execution state to `waiting_for_human`.

4. `GET /workflows/{id}/export` produces a ZIP with an executable Python runner,
graph definition, input example, environment template and separate ML-node
modules. Secrets are never exported.


Agent prompts, skills, diagnostic models and optimizer extension points are documented in [PLANNING_AGENTS.md](PLANNING_AGENTS.md).
