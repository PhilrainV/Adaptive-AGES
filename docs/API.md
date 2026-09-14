# Adaptive-AGES API

Base URL: `/api/v1`. Protected routes use `Authorization: Bearer <token>`.

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/auth/register` | Create a user and initialize the adaptive profile |
| POST | `/auth/login` | Issue an access token |
| GET | `/agents` | List the current user's execution subjects |
| POST | `/agents` | Register an LLM, ML, human, or tool subject and its capability vector |
| POST | `/tasks/understand` | Convert a natural-language goal to a task graph |
| POST | `/workflows/plan` | Match subjects and generate an explainable workflow |
| PUT | `/workflows/{id}` | Persist edited nodes, edges, prompts, code and review criteria |
| POST | `/workflows/{id}/execute` | Execute the saved graph with a task input |
| GET | `/workflows/{id}/export` | Download an independent executable workflow ZIP |
| POST | `/workflows/execute` | Compile and execute a workflow through LangGraph |
| POST | `/executions/{id}/human-feedback` | Submit a human decision or correction and make a paused run resumable |
| GET/PUT | `/settings/model` | Read or save encrypted OpenAI-compatible model configuration |
| POST | `/settings/model/test` | Validate the stored API configuration |
| GET | `/human-assessments/profile` | Read evidence-backed human capability values |
| POST | `/human-assessments/generate` | Generate a test from a system design requirement |
| POST | `/human-assessments/{id}/submit` | Score answers and update the Human Agent profile |

## Core exchange

1. `POST /tasks/understand`

```json
{
  "prompt": "分析学生学习数据，预测学业风险，生成个性化教学建议并由教师复核",
  "constraints": {"max_latency_seconds": 120, "explainable": true}
}
```

2. `POST /workflows/plan` receives the returned task graph, an available capability space, user profile, and optimization weights. It returns nodes, edges, estimated cost, and a `decision_trace`. Every trace entry records the chosen subject, normalized score, reasons, and alternatives.

3. The client may edit and persist the plan with `PUT /workflows/{id}`. `POST /workflows/{id}/execute`
compiles the saved DAG. A human node changes the execution state to `waiting_for_human`.

4. `GET /workflows/{id}/export` produces a ZIP with an executable Python runner, graph definition,
input example, environment template and separate ML-node modules. Secrets are never exported.
