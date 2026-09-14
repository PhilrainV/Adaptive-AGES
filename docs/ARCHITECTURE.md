# Architecture

## Design principle

Adaptive-AGES treats an LLM, a machine-learning model, and a human as first-class execution subjects. The planner never binds a task type directly to a fixed executor. It first creates a task graph, expresses every subtask in the shared capability space, ranks available subjects, then compiles the selected graph for execution.

```mermaid
flowchart TD
  U["Natural-language task"] --> A1["1. Problem Analysis Agent"]
  A1 --> A2["2. Test Generation Agent"]
  A2 --> Q["User answers"]
  Q --> A3["3. Ability Diagnosis Agent"]
  A3 --> A4["4. Capability Planning Agent"]
  A4 --> G["Dynamic Workflow"]
  G --> E["LangGraph Engine"]
  E --> F["Feedback & Capability Learning"]
  F --> A3
```

## Module boundaries

| Layer | Responsibility | Extension point |
| --- | --- | --- |
| Problem Analysis Agent | Goal extraction, decomposition, dependency and risk analysis | Structured-output LLM adapter or domain-specific decomposer |
| Test Generation Agent | Generate contextual evidence tasks from the current problem and graph | Adaptive item bank, IRT, or multimodal assessment |
| Ability Diagnosis Agent | Score answers and map evidence to shared planning capabilities | Bayesian estimation, confidence calibration, or longitudinal evidence |
| Capability Planning Agent | Inject the fresh Human capability vector before multi-objective matching | Neural ranker, constraint solver, bandit or reinforcement learner |
| Capability Modeling | Versioned capability vectors, evidence and confidence | Benchmark import, embeddings, graph features |
| Executor Registry | One `execute(task, context)` contract across subject types | Custom LLM, PyTorch, sklearn, API and human executors |
| LangGraph Engine | DAG compilation, state propagation and pause/resume | Checkpointer, parallel branches, conditional edges |
| Capability Learning | Updates from outcomes, correction and disagreement | Bayesian calibration, online learning, drift detection |

## Matching objective

For task requirement vector `r` and subject capability vector `c`, the MVP uses:

`score = α · cosine(r,c) + β · reliability − γ · cost − δ · latency + risk_bonus`

The scoring contract is independent from the algorithm. A neural matcher or graph matcher can replace cosine similarity while retaining the same planning and explanation interfaces.

## Persistence model

| Table | Main role |
| --- | --- |
| `users` | Identity and user-adaptive profile |
| `agents` / `agent_capabilities` | LLM and tool subjects plus evidence-backed capability values |
| `ml_models` / `model_capabilities` | ML artifact registry, signatures and capabilities |
| `human_profiles` | Roles, availability, decision history and evolving human capability |
| `human_assessments` | Task-derived questions, submitted evidence and scored capability vectors |
| `model_settings` | Per-user provider settings and encrypted model credentials |
| `tasks` / `task_graphs` | Original goal, constraints and decomposed DAG |
| `workflows` | Versioned dynamic graph and explainable decision trace |
| `workflow_executions` | Runtime state, node outputs and pause/resume status |
| `feedback` | Human correction, rating and rationale for later capability learning |

## Phase boundaries

- Phase 1 is represented by identity, agent registry, capability matching, task planning, the React Flow studio, and LangGraph execution.
- Phase 2 adds production ML artifact loading, human work queues, availability-aware routing, and behavioral user-profile updates.
- Phase 3 consumes execution evidence to calibrate capability scores, quantify contribution and expose counterfactual route explanations.
