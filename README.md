# Adaptive-AGES

Adaptive Generalized Execution System is an extensible platform for automatically composing LLM agents, machine-learning models, tools, and human experts around complex tasks. Unlike a conventional Agent Builder, its primary flow is **task → capability requirements → subject matching → dynamic workflow → execution feedback**.

The repository contains a production-oriented MVP: a responsive React/TypeScript workspace, React Flow workflow canvas, capability and user-profile views, plus a modular FastAPI backend with PostgreSQL models, Redis/Celery jobs, explainable matching, and LangGraph compilation.

## Implemented MVP

- Natural-language task decomposition into a typed task graph
- Shared capability representation for LLM, ML, human, and tool subjects
- Multi-objective ranking with cost, latency, reliability, risk, and human-in-the-loop policy
- Explainable decision trace with selected and alternative subjects
- Executor registry and LangGraph workflow compiler
- Human-node pause contract and feedback endpoint
- User registration, login, adaptive profile, and agent management APIs
- Interactive studio, workflow execution animation, capability radar, and editable user profile
- PostgreSQL entity model and Redis/Celery worker foundation

## Repository map

```text
app/                    React/Vinext application shell
components/views/       Dashboard, Studio, Capability, and Profile surfaces
components/workflow/    React Flow nodes and canvas
lib/                    Shared frontend domain data
backend/app/api/        FastAPI routes and dependencies
backend/app/models/     SQLAlchemy persistence model
backend/app/services/   Task understanding, matching, and adaptive planning
backend/app/executors/  Unified executor interface and registry
backend/app/workflow/   LangGraph compiler and runtime
backend/app/workers/    Celery configuration and jobs
docs/                   Architecture and API contract
```

## Run locally

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

The interactive API contract is available at `http://localhost:8000/docs`.

## Production hardening before public use

Replace the development JWT secret, add Alembic migrations to the release process, configure an object store/model registry for ML artifacts, add LangGraph checkpoint persistence, and route all credentials through a secret manager. Human decisions should be authorized by task/workspace membership and logged with immutable audit metadata.

See [Architecture](docs/ARCHITECTURE.md) and [API](docs/API.md) for extension details.
