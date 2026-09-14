# Adaptive-AGES

Adaptive Generalized Execution System is an extensible platform for automatically composing LLM agents, machine-learning models, tools, and human experts around complex tasks. Unlike a conventional Agent Builder, its primary flow is **problem analysis → task-specific ability test → human ability diagnosis → capability-aware planning → execution feedback**.

The repository contains a production-oriented MVP: a responsive React/TypeScript workspace, React Flow workflow canvas, capability and user-profile views, plus a modular FastAPI backend with PostgreSQL models, Redis/Celery jobs, explainable matching, and LangGraph compilation.

## Implemented MVP

- Natural-language task decomposition into a typed task graph
- Shared capability representation for LLM, ML, human, and tool subjects
- Multi-objective ranking with cost, latency, reliability, risk, and human-in-the-loop policy
- Explainable decision trace with selected and alternative subjects
- Executor registry and LangGraph workflow compiler
- Human-node pause contract and feedback endpoint
- User registration, login, adaptive profile, and agent management APIs
- Interactive studio backed by the FastAPI planning API (no static workflow simulation)
- Add, delete, move and connect nodes; edit LLM prompts, ML Python and human-review criteria
- Encrypted per-user OpenAI-compatible model settings and connection test
- Real LangGraph execution plus a standalone executable ZIP export
- An assessment-first planning session implemented by four separate agents: problem analysis, test generation, ability diagnosis, and capability-aware planning
- Dashboard task navigation, pending-assessment recovery, and owner-scoped cascading deletion
- PostgreSQL entity model and Redis/Celery worker foundation

## Repository map

```text
app/                    React/Vinext application shell
components/views/       Dashboard, Studio, Capability, and Profile surfaces
components/workflow/    React Flow nodes and canvas
lib/                    Shared frontend domain data
backend/app/api/        FastAPI routes and dependencies
backend/app/agents/     Four specialized agents in the assessment-first planning pipeline
backend/app/models/     SQLAlchemy persistence model
backend/app/services/   Task understanding, matching, and adaptive planning
backend/app/executors/  Unified executor interface and registry
backend/app/workflow/   LangGraph compiler and runtime
backend/app/workers/    Celery configuration and jobs
docs/                   Architecture and API contract
```

## One-command Docker deployment (recommended)

Requirements: Docker Desktop with Docker Compose.

```bash
cp .env.example .env
docker compose -p adaptive-ages up -d --build
```

Open the complete platform at `http://localhost:8080`. API documentation is available at
`http://localhost:8080/docs` (or directly at `http://localhost:8000/docs`). The single `8080`
gateway serves the frontend and proxies `/api/v1` to FastAPI, so the same deployment works behind
a domain or tunnel without changing browser API URLs.

Stop services without deleting data:

```bash
docker compose -p adaptive-ages down
```

### Temporary public HTTPS address

The optional `public` profile starts a Cloudflare Quick Tunnel:

```bash
docker compose -p adaptive-ages --profile public up -d --build
docker compose -p adaptive-ages logs -f tunnel
```

Copy the `https://...trycloudflare.com` URL printed in the tunnel log. This URL is temporary and
changes when the tunnel container is recreated. Anyone with the URL can reach the development
workspace, so do not use Quick Tunnel for sensitive or permanent production workloads.

## Run services separately

Requirements: Node.js 22+, pnpm, Python 3.11+, PostgreSQL 16, and Redis 7.

```bash
cp .env.example .env
pnpm install
pnpm dev
```

In another terminal:

```bash
docker compose up -d postgres redis
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Celery worker:

```bash
cd backend
celery -A app.workers.celery_app:celery_app worker --loglevel=info
```

For this mode, open the frontend at `http://localhost:5173` and API documentation at
`http://localhost:8000/docs`.

## Workflow execution and export

`运行并导出` first saves the edited graph, executes it with LangGraph, then downloads a ZIP bundle.
The bundle contains `workflow.json`, `run_workflow.py`, `input.json`, model environment examples,
requirements, and one editable Python module per ML node. API keys are never included in exports.

Custom ML code is intentionally not executed inside the platform API process. It is executed only
when the owner runs the exported package, preventing arbitrary user code from compromising the
orchestration service.

## Production hardening before public use

Replace the development JWT secret, add Alembic migrations to the release process, configure an object store/model registry for ML artifacts, add LangGraph checkpoint persistence, and route all credentials through a secret manager. Human decisions should be authorized by task/workspace membership and logged with immutable audit metadata.

See [Architecture](docs/ARCHITECTURE.md) and [API](docs/API.md) for extension details.
