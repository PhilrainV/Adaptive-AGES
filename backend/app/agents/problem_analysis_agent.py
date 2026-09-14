from app.schemas.domain import TaskGraph
from app.services.task_understanding import TaskUnderstandingEngine


class ProblemAnalysisAgent:
    """Agent 1: turns the user's problem into a capability-labelled task graph."""

    name = "problem_analysis_agent"

    def __init__(self, engine: TaskUnderstandingEngine | None = None):
        self.engine = engine or TaskUnderstandingEngine()

    async def run(self, prompt: str, model_config: dict | None = None) -> TaskGraph:
        return await self.engine.understand_async(prompt, model_config)
