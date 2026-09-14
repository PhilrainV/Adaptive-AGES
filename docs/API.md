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
| POST | `/workflows/execute` | Compile and execute a workflow through LangGraph |
| POST | `/executions/{id}/human-feedback` | Submit a human decision or correction and make a paused run resumable |

## Core exchange

1. `POST /tasks/understand`

```json
{
  "prompt": "分析学生学习数据，预测学业风险，生成个性化教学建议并由教师复核",
  "constraints": {"max_latency_seconds": 120, "explainable": true}
}
```

2. `POST /workflows/plan` receives the returned task graph, an available capability space, user profile, and optimization weights. It returns nodes, edges, estimated cost, and a `decision_trace`. Every trace entry records the chosen subject, normalized score, reasons, and alternatives.

3. `POST /workflows/execute` compiles the generated DAG. A human node changes the execution state to `waiting_for_human`; feedback can then be posted without rebuilding the workflow.
