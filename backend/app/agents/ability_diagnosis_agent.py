from app.services.human_assessment import HumanCapabilityAssessmentService


class AbilityDiagnosisAgent:
    """Agent 3: converts test evidence into user and planner capability vectors."""

    name = "ability_diagnosis_agent"

    def __init__(self, service: HumanCapabilityAssessmentService | None = None):
        self.service = service or HumanCapabilityAssessmentService()

    def run(self, questions: list[dict], answers: dict[str, int]) -> dict:
        result = self.service.score(questions, answers)
        raw = result["capability"]
        planning_capability = {
            "reasoning": self._average(raw, "workflow_design", "judgement", "ai_literacy"),
            "prediction": self._average(raw, "domain_knowledge", "programming"),
            "generation": self._average(raw, "domain_knowledge", "judgement"),
            "interpretation": self._average(
                raw, "ai_literacy", "domain_knowledge", "judgement"
            ),
            "domain_knowledge": raw.get("domain_knowledge", 0.5),
            "human_judgement": raw.get("judgement", 0.5),
            "data_processing": self._average(raw, "programming", "workflow_design"),
        }
        answered = result["answered"]
        total = result["total"]
        confidence = round(min(1.0, answered / total), 3) if total else 0.0
        weakest = sorted(raw, key=raw.get)[:2]
        return {
            **result,
            "planning_capability": planning_capability,
            "confidence": confidence,
            "weakest_dimensions": weakest,
        }

    @staticmethod
    def _average(capability: dict[str, float], *dimensions: str) -> float:
        values = [capability.get(dimension, 0.5) for dimension in dimensions]
        return round(sum(values) / len(values), 3)
