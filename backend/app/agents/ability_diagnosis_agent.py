"""Agent 3: cognitive diagnosis based on a Bayesian DINA model.

This file is the main research extension point for human ability measurement.
Replace ``BayesianDINA`` or inject another estimator into ``AbilityDiagnosisAgent``
to compare DINA, G-DINA, NIDA, IRT or neural cognitive diagnosis methods while
keeping the orchestration API unchanged.
"""

from __future__ import annotations

from itertools import product
from math import exp, log
from typing import Any, Protocol

from app.schemas.domain import AgentRuntimeConfig, TaskGraph

DIMENSIONS = [
    "programming",
    "ai_literacy",
    "domain_knowledge",
    "workflow_design",
    "judgement",
]

PLANNING_CAPABILITY_MAP: dict[str, tuple[str, ...]] = {
    "reasoning": ("workflow_design", "judgement", "ai_literacy"),
    "prediction": ("domain_knowledge", "programming"),
    "generation": ("domain_knowledge", "judgement", "ai_literacy"),
    "interpretation": ("ai_literacy", "domain_knowledge", "judgement"),
    "domain_knowledge": ("domain_knowledge",),
    "human_judgement": ("judgement",),
    "data_processing": ("programming", "workflow_design"),
}


class CognitiveDiagnosisEstimator(Protocol):
    name: str

    def estimate(
        self,
        questions: list[dict[str, Any]],
        answers: dict[str, int],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class BayesianDINA:
    """Exact posterior estimation over the 2^K latent mastery classes.

    For item i and mastery pattern alpha, eta_i is one only when every skill
    required by the Q-matrix row is mastered. P(X_i=1|alpha) is then 1-slip_i;
    otherwise it is guess_i. With five attributes, exact enumeration contains
    only 32 classes and is deterministic, transparent and easy to validate.
    """

    name = "bayesian_dina"

    def estimate(
        self,
        questions: list[dict[str, Any]],
        answers: dict[str, int],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = parameters or {}
        dimensions = list(parameters.get("dimensions") or DIMENSIONS)
        prior_value = parameters.get("mastery_prior", .5)
        priors = {
            dimension: self._clamp(
                prior_value.get(dimension, .5) if isinstance(prior_value, dict) else float(prior_value),
                .05,
                .95,
            )
            for dimension in dimensions
        }
        default_guess = self._clamp(float(parameters.get("default_guess", .22)), .01, .49)
        default_slip = self._clamp(float(parameters.get("default_slip", .12)), .01, .49)
        indexed = {dimension: index for index, dimension in enumerate(dimensions)}
        patterns = list(product((0, 1), repeat=len(dimensions)))
        log_posteriors: list[float] = []

        for pattern in patterns:
            log_probability = 0.0
            for dimension, mastery in zip(dimensions, pattern, strict=True):
                prior = priors[dimension]
                log_probability += log(prior if mastery else 1 - prior)
            for item in questions:
                components = [
                    component
                    for component in (item.get("knowledge_components") or [item.get("dimension")])
                    if component in indexed
                ]
                eta = bool(components) and all(pattern[indexed[component]] for component in components)
                guess = self._clamp(float(item.get("guess", default_guess)), .01, .49)
                slip = self._clamp(float(item.get("slip", default_slip)), .01, .49)
                probability_correct = 1 - slip if eta else guess
                observed = answers.get(item["id"]) == item.get("correct_index")
                probability = probability_correct if observed else 1 - probability_correct
                log_probability += log(max(probability, 1e-12))
            log_posteriors.append(log_probability)

        maximum = max(log_posteriors)
        unnormalised = [exp(value - maximum) for value in log_posteriors]
        normaliser = sum(unnormalised)
        posterior = [value / normaliser for value in unnormalised]
        mastery = {
            dimension: round(
                sum(probability for pattern, probability in zip(patterns, posterior, strict=True) if pattern[indexed[dimension]]),
                4,
            )
            for dimension in dimensions
        }
        entropy = -sum(probability * log(probability) for probability in posterior if probability > 0)
        normalised_entropy = entropy / log(len(patterns)) if len(patterns) > 1 else 0
        q_coverage = {
            dimension: sum(
                dimension in (item.get("knowledge_components") or [item.get("dimension")])
                for item in questions
            )
            for dimension in dimensions
        }
        coverage_score = min(1.0, min(q_coverage.values(), default=0) / 2)
        separation = sum(
            1 - float(item.get("slip", default_slip)) - float(item.get("guess", default_guess))
            for item in questions
        ) / max(1, len(questions))
        confidence = self._clamp(
            .55 * (1 - normalised_entropy) + .25 * coverage_score + .2 * separation,
            0,
            1,
        )
        top_classes = sorted(zip(patterns, posterior, strict=True), key=lambda pair: pair[1], reverse=True)[:5]
        return {
            "method": self.name,
            "model": {
                "type": "DINA",
                "dimensions": dimensions,
                "latent_classes": len(patterns),
                "mastery_prior": priors,
            },
            "capability": mastery,
            "posterior_mastery": mastery,
            "confidence": round(confidence, 4),
            "posterior_entropy": round(normalised_entropy, 4),
            "q_matrix_coverage": q_coverage,
            "top_mastery_patterns": [
                {
                    "pattern": dict(zip(dimensions, pattern, strict=True)),
                    "probability": round(probability, 4),
                }
                for pattern, probability in top_classes
            ],
            "item_analysis": self._item_analysis(questions, answers, patterns, posterior, indexed, default_guess, default_slip),
            "answered": len(answers),
            "total": len(questions),
        }

    @staticmethod
    def _item_analysis(
        questions: list[dict[str, Any]],
        answers: dict[str, int],
        patterns: list[tuple[int, ...]],
        posterior: list[float],
        indexed: dict[str, int],
        default_guess: float,
        default_slip: float,
    ) -> list[dict[str, Any]]:
        result = []
        for item in questions:
            components = [component for component in (item.get("knowledge_components") or [item.get("dimension")]) if component in indexed]
            guess = float(item.get("guess", default_guess))
            slip = float(item.get("slip", default_slip))
            predicted = 0.0
            for pattern, probability in zip(patterns, posterior, strict=True):
                eta = bool(components) and all(pattern[indexed[component]] for component in components)
                predicted += probability * (1 - slip if eta else guess)
            observed = answers.get(item["id"]) == item.get("correct_index")
            result.append({
                "item_id": item["id"],
                "knowledge_components": components,
                "observed_correct": observed,
                "posterior_probability_correct": round(predicted, 4),
                "residual": round((1 if observed else 0) - predicted, 4),
            })
        return result

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


class AbilityDiagnosisAgent:
    """Convert diagnostic evidence into human and planner capability vectors."""

    name = "ability_diagnosis_agent"
    model_version = "dina-1.0"

    def __init__(self, estimator: CognitiveDiagnosisEstimator | None = None):
        self.estimator = estimator or BayesianDINA()

    def run(
        self,
        questions: list[dict[str, Any]],
        answers: dict[str, int],
        task_graph: TaskGraph | None = None,
        agent_config: AgentRuntimeConfig | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._normalise_config(agent_config)
        result = self.estimator.estimate(questions, answers, config.get("parameters"))
        raw = result["capability"]
        planning_capability = {
            capability: round(sum(raw.get(dimension, .5) for dimension in dimensions) / len(dimensions), 4)
            for capability, dimensions in PLANNING_CAPABILITY_MAP.items()
        }
        task_weights = self._task_dimension_weights(task_graph)
        denominator = sum(task_weights.values()) or 1
        overall = sum(raw.get(dimension, .5) * weight for dimension, weight in task_weights.items()) / denominator
        mastered_threshold = float(config.get("parameters", {}).get("mastered_threshold", .7))
        developing_threshold = float(config.get("parameters", {}).get("developing_threshold", .4))
        status = {
            dimension: "mastered" if probability >= mastered_threshold else "developing" if probability >= developing_threshold else "not_mastered"
            for dimension, probability in raw.items()
        }
        weakest = sorted(raw, key=raw.get)[:2]
        return {
            **result,
            "agent": self.name,
            "model_version": self.model_version,
            "overall": round(overall, 4),
            "planning_capability": planning_capability,
            "mastery_status": status,
            "task_dimension_weights": task_weights,
            "weakest_dimensions": weakest,
            "diagnostic_summary": {
                "mastered": [dimension for dimension, value in status.items() if value == "mastered"],
                "developing": [dimension for dimension, value in status.items() if value == "developing"],
                "not_mastered": [dimension for dimension, value in status.items() if value == "not_mastered"],
            },
        }

    @staticmethod
    def _task_dimension_weights(task_graph: TaskGraph | None) -> dict[str, float]:
        if not task_graph:
            return {dimension: 1.0 for dimension in DIMENSIONS}
        mapping = {
            "programming": ("prediction", "data_processing"),
            "ai_literacy": ("reasoning", "generation", "interpretation"),
            "domain_knowledge": ("domain_knowledge",),
            "workflow_design": ("reasoning", "data_processing"),
            "judgement": ("human_judgement", "interpretation"),
        }
        totals = {dimension: 0.0 for dimension in DIMENSIONS}
        for item in task_graph.subtasks:
            requirement = item.requirement.model_dump()
            for dimension, capabilities in mapping.items():
                totals[dimension] += sum(requirement[name] for name in capabilities) / len(capabilities)
        return {dimension: round(max(.1, value), 4) for dimension, value in totals.items()}

    @staticmethod
    def _normalise_config(config: AgentRuntimeConfig | dict[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        return config.model_dump() if isinstance(config, AgentRuntimeConfig) else dict(config)
