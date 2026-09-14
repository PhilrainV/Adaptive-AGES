import hashlib
from app.schemas.domain import CapabilityRequirement, Subtask, TaskGraph


class TaskUnderstandingEngine:
    """Converts natural-language goals into an executable task DAG.

    The deterministic fallback is intentionally explicit and testable. A production
    deployment can inject an LLM structured-output adapter without changing callers.
    """

    def understand(self, prompt: str) -> TaskGraph:
        task_id = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        text = prompt.lower()
        subtasks: list[Subtask] = []

        if any(word in text for word in ["数据", "data", "成绩", "csv"]):
            subtasks.append(Subtask(id="prepare", name="数据准备", description="校验、清洗并构造可用特征", task_type="data_processing", requirement=CapabilityRequirement(data_processing=.95, prediction=.35), risk=.2))
        if any(word in text for word in ["预测", "分类", "风险", "forecast", "predict"]):
            subtasks.append(Subtask(id="predict", name="模型预测", description="对结构化特征执行预测并输出不确定性", task_type="prediction", requirement=CapabilityRequirement(prediction=.95, interpretation=.65), dependencies=[subtasks[-1].id] if subtasks else [], risk=.45))
        if any(word in text for word in ["建议", "解释", "报告", "生成", "explain", "generate"]):
            subtasks.append(Subtask(id="explain", name="解释与建议生成", description="将分析结果转化为符合领域约束的解释", task_type="generation", requirement=CapabilityRequirement(reasoning=.9, generation=.95, interpretation=.9, domain_knowledge=.7), dependencies=[subtasks[-1].id] if subtasks else [], risk=.55))
        if any(word in text for word in ["审核", "复核", "教师", "专家", "责任", "human"]):
            subtasks.append(Subtask(id="review", name="人类专业复核", description="结合真实情境校准输出并承担最终判断责任", task_type="human_review", requirement=CapabilityRequirement(domain_knowledge=.95, human_judgement=.98, interpretation=.75), dependencies=[subtasks[-1].id] if subtasks else [], risk=.85))
        if not subtasks:
            subtasks.append(Subtask(id="reason", name="任务推理与生成", description="分析任务并生成结构化结果", task_type="reasoning", requirement=CapabilityRequirement(reasoning=.9, generation=.8), risk=.4))

        breadth = sum(1 for s in subtasks for value in s.requirement.model_dump().values() if value >= .7)
        complexity = min(.98, .18 + len(subtasks) * .14 + breadth * .035)
        return TaskGraph(task_id=task_id, goal=prompt, complexity=round(complexity, 3), subtasks=subtasks)
