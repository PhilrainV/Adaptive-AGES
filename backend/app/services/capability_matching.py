from math import sqrt
from app.schemas.domain import CapabilitySubject, MatchResult, Subtask


class CapabilityMatcher:
    def _cosine(self, requirement: dict[str, float], capability: dict[str, float]) -> float:
        keys = set(requirement) | set(capability)
        dot = sum(requirement.get(k, 0) * capability.get(k, 0) for k in keys)
        nr = sqrt(sum(requirement.get(k, 0) ** 2 for k in keys))
        nc = sqrt(sum(capability.get(k, 0) ** 2 for k in keys))
        return dot / (nr * nc) if nr and nc else 0.0

    def rank(self, subtask: Subtask, subjects: list[CapabilitySubject], weights: dict[str, float], user_profile: dict[str, float] | None = None) -> list[MatchResult]:
        requirement = subtask.requirement.model_dump()
        ranked: list[MatchResult] = []
        for subject in subjects:
            similarity = self._cosine(requirement, subject.capability)
            human_bonus = .08 if subtask.risk >= .7 and subject.subject_type.value == "human" else 0
            score = weights["fit"] * similarity + weights["reliability"] * subject.reliability - weights["cost"] * subject.cost - weights["latency"] * subject.latency + human_bonus
            key_dims = sorted(requirement, key=requirement.get, reverse=True)[:2]
            explanation = f"{subtask.name}最需要{'、'.join(key_dims)}；{subject.name}在这些维度上的综合适配度为{similarity:.0%}，结合可靠性、成本与时延后排名当前第一。"
            ranked.append(MatchResult(subject_id=subject.id, subject_name=subject.name, subject_type=subject.subject_type, score=round(max(0, min(1, score)), 4), similarity=round(similarity, 4), explanation=explanation))
        ranked.sort(key=lambda item: item.score, reverse=True)
        for item in ranked:
            item.alternatives = [{"id": alt.subject_id, "name": alt.subject_name, "score": alt.score} for alt in ranked if alt.subject_id != item.subject_id][:3]
        return ranked
