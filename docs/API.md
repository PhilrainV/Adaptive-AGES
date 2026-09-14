# Adaptive-AGES API

Base URL: `/api/v1`. Protected routes use `Authorization: Bearer <token>`.

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/auth/register` | Create a user and initialize the adaptive profile |
| POST | `/auth/login` | Issue an access token |
| GET | `/agents` | List the current user's execution subjects |
| POST | `/agents` | Register an LLM, ML, human, or tool subject and its capability vector |
| GET | `/dashboard` | Return real task, workflow, execution, and aggregate workspace data |
| POST | `/planning-sessions/start` | Analyse a problem and generate a task-specific ability test |
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

1. `POST /planning-sessions/start`

```json
{
  "prompt": "分析学生学习数据，预测学业风险，生成个性化教学建议并由教师复核",
  "constraints": {"max_latency_seconds": 120, "explainable": true}
}
```

The problem-analysis agent returns a task graph and the test-generation agent returns task-specific questions. No workflow is created yet.

2. After the user answers every question, `POST /planning-sessions/{id}/diagnose` runs only the ability-diagnosis agent. It persists the evidence-backed diagnosis and returns it to the client. No workflow exists at this point.

3. After the user reviews the result and explicitly confirms planning, `POST /planning-sessions/{id}/plan` passes the stored task graph and diagnosis to the capability-planning agent. Only this request creates the workflow. Every trace entry records the chosen subject, normalized score, reasons, and alternatives.

4. The client may edit and persist the plan with `PUT /workflows/{id}`. `POST /workflows/{id}/execute`
compiles the saved DAG. A human node changes the execution state to `waiting_for_human`.

5. `GET /workflows/{id}/export` produces a ZIP with an executable Python runner, graph definition,
input example, environment template and separate ML-node modules. Secrets are never exported.
