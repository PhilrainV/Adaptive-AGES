from __future__ import annotations

from typing import Any, ClassVar

from app.schemas.domain import TaskGraph


class HumanCapabilityAssessmentService:
    """Deterministic item bank used when an LLM is unavailable.

    Scoring remains only for the legacy standalone assessment API. Automatic
    planning uses ``AbilityDiagnosisAgent`` and its Bayesian DINA estimator.
    """

    dimensions: ClassVar[list[str]] = [
        "programming",
        "ai_literacy",
        "domain_knowledge",
        "workflow_design",
        "judgement",
    ]

    @staticmethod
    def _item(
        item_id: str,
        dimension: str,
        prompt: str,
        options: list[str],
        correct_index: int,
        *,
        subtask_id: str | None = None,
        difficulty: float = .5,
    ) -> dict[str, Any]:
        return {
            "id": item_id,
            "dimension": dimension,
            "knowledge_components": [dimension],
            "subtask_id": subtask_id,
            "prompt": prompt,
            "options": options,
            "correct_index": correct_index,
            "rationale": "用于该属性的单属性锚题",
            "difficulty": difficulty,
            "discrimination": 1.0,
            "guess": .22,
            "slip": .12,
        }

    def generate(self, requirement: str, task_graph: TaskGraph | None = None) -> list[dict[str, Any]]:
        context = requirement.strip()[:180]
        subtask_ids = [item.id for item in task_graph.subtasks] if task_graph else []

        def sid(index: int) -> str | None:
            return subtask_ids[min(index, len(subtask_ids) - 1)] if subtask_ids else None

        return [
            self._item("q1", "workflow_design", f"针对“{context}”，哪种拆分最利于追踪错误和责任？", ["全部交给一个模型", "拆成输入校验、分析、生成与复核并记录依赖", "只保留最终输出", "随机选择执行主体"], 1, subtask_id=sid(0), difficulty=.35),
            self._item("q2", "workflow_design", "并行节点汇合前首先需要明确什么？", ["屏幕分辨率", "汇合条件、缺失结果策略和数据契约", "开发者昵称", "节点颜色"], 1, subtask_id=sid(1), difficulty=.55),
            self._item("q3", "ai_literacy", "LLM 给出流畅但缺少证据的结论，最合适的处理是什么？", ["直接采用", "提高 temperature", "要求引用上游证据并增加验证节点", "删除日志"], 2, subtask_id=sid(2), difficulty=.42),
            self._item("q4", "ai_literacy", "比较两个候选智能体时，哪组指标更完整？", ["只看模型规模", "能力匹配、可靠性、成本、时延与风险", "只看价格", "只看响应长度"], 1, subtask_id=sid(1), difficulty=.48),
            self._item("q5", "programming", "自定义 ML 节点要被工作流稳定调用，最关键的接口约束是什么？", ["固定输入输出结构并处理异常", "代码越长越好", "必须使用深度学习", "不声明依赖"], 0, subtask_id=sid(0), difficulty=.48),
            self._item("q6", "programming", "一个节点偶发失败且不能重复产生副作用，首先应加入什么？", ["更鲜艳的界面", "幂等键、异常处理和有界重试", "更高 temperature", "删除执行记录"], 1, subtask_id=sid(0), difficulty=.65),
            self._item("q7", "domain_knowledge", f"判断“{context}”的结果是否可用，最重要的领域证据是什么？", ["界面颜色", "模型名称", "任务目标、数据含义和实际使用约束", "输出字数"], 2, subtask_id=sid(2), difficulty=.45),
            self._item("q8", "domain_knowledge", "领域指标与实际目标冲突时，应如何处理？", ["只保留最高的技术指标", "回到领域目标核验指标含义并记录取舍", "随机选择", "隐藏冲突"], 1, subtask_id=sid(2), difficulty=.62),
            self._item("q9", "judgement", "哪种情况最应该触发人类专家介入？", ["低风险格式转换", "缓存读取", "涉及权益且模型分歧显著的决策", "字段重命名"], 2, subtask_id=sid(3), difficulty=.4),
            self._item("q10", "judgement", f"对“{context}”的最终输出，人工复核最有价值的行为是什么？", ["只点击通过", "依据证据修订并记录理由", "清除执行轨迹", "隐藏不确定性"], 1, subtask_id=sid(3), difficulty=.6),
        ]

    def score(self, questions: list[dict[str, Any]], answers: dict[str, int]) -> dict[str, Any]:
        totals = {dimension: 0 for dimension in self.dimensions}
        correct = {dimension: 0 for dimension in self.dimensions}
        for question in questions:
            dimension = question["dimension"]
            totals[dimension] += 1
            if answers.get(question["id"]) == question["correct_index"]:
                correct[dimension] += 1
        capability = {
            dimension: round(correct[dimension] / totals[dimension], 3) if totals[dimension] else .5
            for dimension in self.dimensions
        }
        overall = round(sum(capability.values()) / len(capability), 3)
        return {"capability": capability, "overall": overall, "answered": len(answers), "total": len(questions)}

    @staticmethod
    def public_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hidden = {"correct_index", "rationale", "guess", "slip", "discrimination"}
        return [{key: value for key, value in question.items() if key not in hidden} for question in questions]
