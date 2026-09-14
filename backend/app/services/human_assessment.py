from typing import Any, ClassVar


class HumanCapabilityAssessmentService:
    """Creates a task-grounded assessment and converts evidence into capabilities."""

    dimensions: ClassVar[list[str]] = ["programming", "ai_literacy", "domain_knowledge", "workflow_design", "judgement"]

    def generate(self, requirement: str) -> list[dict[str, Any]]:
        context = requirement.strip()[:180]
        return [
            {
                "id": "q1", "dimension": "workflow_design",
                "prompt": f"针对“{context}”，哪种拆分方式最有利于追踪错误和责任？",
                "options": ["将全部工作交给一个模型", "拆成输入校验、分析、生成和复核并记录依赖", "只保留最终输出", "随机选择执行主体"],
                "correct_index": 1,
            },
            {
                "id": "q2", "dimension": "ai_literacy",
                "prompt": "LLM 生成了语言流畅但缺少证据的结论，最合适的处理是什么？",
                "options": ["直接采用", "提高 temperature", "要求引用上游证据并增加验证节点", "删除日志"],
                "correct_index": 2,
            },
            {
                "id": "q3", "dimension": "programming",
                "prompt": "自定义 ML 节点需要被工作流稳定调用，最关键的接口约束是什么？",
                "options": ["固定输入输出结构并处理异常", "代码行数尽量多", "必须使用深度学习", "不声明依赖"],
                "correct_index": 0,
            },
            {
                "id": "q4", "dimension": "domain_knowledge",
                "prompt": f"在“{context}”中评估结果是否可用，最重要的领域证据是什么？",
                "options": ["界面颜色", "模型名称", "任务目标、数据含义与实际使用约束", "输出字数"],
                "correct_index": 2,
            },
            {
                "id": "q5", "dimension": "judgement",
                "prompt": "哪种情况最应该触发人类专家介入？",
                "options": ["低风险格式转换", "缓存读取", "涉及权益且模型分歧显著的决策", "字段重命名"],
                "correct_index": 2,
            },
            {
                "id": "q6", "dimension": "ai_literacy",
                "prompt": "比较两个候选智能体时，哪组指标更完整？",
                "options": ["只看模型规模", "能力匹配、可靠性、成本、时延与风险", "只看价格", "只看响应长度"],
                "correct_index": 1,
            },
            {
                "id": "q7", "dimension": "workflow_design",
                "prompt": "并行节点汇合前应当明确什么？",
                "options": ["屏幕分辨率", "汇合条件、缺失结果策略和数据契约", "开发者昵称", "节点颜色"],
                "correct_index": 1,
            },
            {
                "id": "q8", "dimension": "judgement",
                "prompt": f"对“{context}”的最终输出，人工复核最有价值的行为是什么？",
                "options": ["只点击通过", "依据具体证据修订并记录理由", "清除执行轨迹", "隐藏不确定性"],
                "correct_index": 1,
            },
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
        return [{key: value for key, value in question.items() if key != "correct_index"} for question in questions]
