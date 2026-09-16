import hashlib

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.schemas.domain import (
    CapabilityRequirement,
    ExecutionMode,
    IterationPolicy,
    Subtask,
    TaskGraph,
)


class GeneratedSubtask(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    requirement: CapabilityRequirement
    input_contract: list[str] = Field(default_factory=list)
    output_contract: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    runtime_hints: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    risk: float = Field(default=.4, ge=0, le=1)


class GeneratedBreakdown(BaseModel):
    subtasks: list[GeneratedSubtask]


class TaskUnderstandingEngine:
    """Converts natural-language goals into an executable task DAG.

    The deterministic fallback is intentionally explicit and testable. A production
    deployment can inject an LLM structured-output adapter without changing callers.
    """

    def understand(self, prompt: str) -> TaskGraph:
        task_id = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        text = prompt.lower()
        subtasks: list[Subtask] = []

        def previous() -> list[str]:
            return [subtasks[-1].id] if subtasks else []

        learning_assessment = any(word in text for word in ["知识水平", "薄弱知识", "薄弱点", "掌握情况", "学情", "能力水平", "学习水平"])
        practice_generation = any(word in text for word in ["练习", "训练题", "习题", "学习任务"])
        summary_generation = any(word in text for word in ["总结", "最终报告", "总结报告"])
        evaluation = any(word in text for word in ["检查学习效果", "检查效果", "学习效果", "是否达标", "达到目标", "效果评价"])
        human_practice = practice_generation and any(
            word in text for word in ["完成每个练习", "完成练习", "学生完成", "用户完成", "作答"]
        )
        emotion_aware = any(
            word in text
            for word in ["情绪", "心情", "焦虑", "沮丧", "挫败", "安抚", "emotion", "mood"]
        )
        if learning_assessment and practice_generation:
            return self._learning_improvement_graph(
                prompt,
                task_id,
                evaluation=evaluation,
                human_practice=human_practice,
                summary_generation=summary_generation,
                iterative=evaluation and any(
                    word in text for word in ["继续调整", "重新生成", "直到", "循环", "迭代"]
                ),
                emotion_aware=emotion_aware,
            )

        if any(word in text for word in ["数据", "data", "成绩", "csv"]):
            subtasks.append(Subtask(id="prepare", name="数据准备", description="校验、清洗并构造可用特征", task_type="data_processing", requirement=CapabilityRequirement(data_processing=.95, prediction=.35), risk=.2))

        if learning_assessment:
            subtasks.append(Subtask(
                id="diagnose-level",
                name="学习水平诊断",
                description="根据已有表现识别已掌握内容、薄弱知识点与诊断置信度",
                task_type="prediction",
                requirement=CapabilityRequirement(prediction=.82, interpretation=.86, domain_knowledge=.72, data_processing=.62),
                dependencies=previous(),
                risk=.45,
            ))
        elif any(word in text for word in ["预测", "分类", "风险", "forecast", "predict"]):
            subtasks.append(Subtask(id="predict", name="模型预测", description="对结构化特征执行预测并输出不确定性", task_type="prediction", requirement=CapabilityRequirement(prediction=.95, interpretation=.65), dependencies=previous(), risk=.45))

        if practice_generation:
            subtasks.append(Subtask(
                id="generate-practice",
                name="薄弱点练习生成",
                description="针对每个薄弱知识点生成难度适配、目标明确的练习任务",
                task_type="generation",
                requirement=CapabilityRequirement(reasoning=.8, generation=.96, interpretation=.82, domain_knowledge=.78),
                dependencies=previous(),
                risk=.35,
                execution_mode=ExecutionMode.CONDITIONAL if learning_assessment else ExecutionMode.SEQUENTIAL,
                entry_condition="weak_points_exist == true" if learning_assessment else None,
            ))

        if human_practice:
            subtasks.append(Subtask(
                id="human-practice",
                name="学习者完成练习",
                description="学习者完成当前练习并提交作答过程、答案与主观困难",
                task_type="human_action",
                requirement=CapabilityRequirement(domain_knowledge=.58, human_judgement=.65),
                dependencies=previous(),
                risk=.15,
            ))

        if evaluation:
            loop_target = "generate-practice" if practice_generation else None
            iterative = bool(loop_target and any(word in text for word in ["继续调整", "重新生成", "直到", "循环", "迭代"] ))
            subtasks.append(Subtask(
                id="evaluate-progress",
                name="学习效果检查",
                description="依据练习结果计算达标状态、薄弱点变化和下一轮调整信号",
                task_type="evaluation",
                requirement=CapabilityRequirement(data_processing=.78, prediction=.68, interpretation=.72),
                dependencies=previous(),
                risk=.3,
                execution_mode=ExecutionMode.ITERATIVE if iterative else ExecutionMode.SEQUENTIAL,
                iteration_policy=IterationPolicy(
                    enabled=iterative,
                    feedback_target_subtask_id=loop_target if iterative else None,
                    condition="goal_reached == false and max_iterations_reached == false",
                    max_iterations=3,
                ),
            ))

        general_generation = any(word in text for word in ["建议", "解释", "生成", "explain", "generate"])
        if general_generation and not practice_generation and not summary_generation:
            subtasks.append(Subtask(id="explain", name="解释与建议生成", description="将分析结果转化为符合领域约束的解释", task_type="generation", requirement=CapabilityRequirement(reasoning=.9, generation=.95, interpretation=.9, domain_knowledge=.7), dependencies=previous(), risk=.55))

        if any(word in text for word in ["审核", "复核", "教师", "专家", "责任", "human"]):
            subtasks.append(Subtask(id="review", name="人类专业复核", description="结合真实情境校准输出并承担最终判断责任", task_type="human_review", requirement=CapabilityRequirement(domain_knowledge=.95, human_judgement=.98, interpretation=.75), dependencies=previous(), risk=.85))

        if summary_generation:
            subtasks.append(Subtask(
                id="summary-report",
                name="学习总结报告",
                description="汇总初始诊断、练习过程、能力变化、达标结果与后续建议",
                task_type="generation",
                requirement=CapabilityRequirement(reasoning=.72, generation=.92, interpretation=.88),
                dependencies=previous(),
                risk=.3,
                execution_mode=ExecutionMode.CONDITIONAL if evaluation else ExecutionMode.SEQUENTIAL,
                entry_condition="goal_reached == true or max_iterations_reached == true" if evaluation else None,
            ))
        if not subtasks:
            subtasks.append(Subtask(id="reason", name="任务推理与生成", description="分析任务并生成结构化结果", task_type="reasoning", requirement=CapabilityRequirement(reasoning=.9, generation=.8), risk=.4))

        breadth = sum(1 for s in subtasks for value in s.requirement.model_dump().values() if value >= .7)
        complexity = min(.98, .18 + len(subtasks) * .14 + breadth * .035)
        return TaskGraph(task_id=task_id, goal=prompt, complexity=round(complexity, 3), subtasks=subtasks, planning_mode="rule")

    @staticmethod
    def _learning_improvement_graph(
        prompt: str,
        task_id: str,
        *,
        evaluation: bool,
        human_practice: bool,
        summary_generation: bool,
        iterative: bool,
        emotion_aware: bool,
    ) -> TaskGraph:
        """Build an executable learning loop with atomic responsibilities.

        Repeated LLM/ML/Tool nodes are intentional: a subject type is a capability
        class, not a singleton role in the workflow.
        """
        subtasks = [
            Subtask(
                id="validate-evidence",
                name="学习证据校验",
                description="校验学生作答记录、知识点标记、得分字段和目标阈值，报告缺失或异常数据",
                task_type="data_processing",
                requirement=CapabilityRequirement(data_processing=.96, interpretation=.45),
                input_contract=["responses[]", "responses[].knowledge_point", "responses[].correct|score", "target_mastery"],
                output_contract=["validated_responses[]", "data_quality.valid", "data_quality.issues[]", "target_mastery"],
                required_skills=["learning_record_validation", "schema_validation"],
                acceptance_criteria=["每条有效记录包含知识点与得分", "异常记录被显式报告而非静默丢弃"],
                runtime_hints={"operation": "validate_learning_evidence"},
                risk=.15,
            ),
            Subtask(
                id="estimate-mastery",
                name="知识掌握度估计",
                description="基于已校验作答证据估计各知识点掌握度、总体水平与估计置信度",
                task_type="prediction",
                requirement=CapabilityRequirement(prediction=.9, data_processing=.82, interpretation=.7),
                dependencies=["validate-evidence"],
                input_contract=["validated_responses[]", "target_mastery"],
                output_contract=["mastery_by_knowledge_point{}", "overall_mastery", "confidence", "sample_size"],
                required_skills=["bayesian_mastery_estimation", "uncertainty_calibration"],
                acceptance_criteria=["每个出现过的知识点均有0到1掌握度", "输出样本量和置信度"],
                runtime_hints={"algorithm": "beta_binomial_mastery"},
                risk=.35,
            ),
        ]
        if emotion_aware:
            subtasks.extend([
                Subtask(
                    id="assess-emotion",
                    name="学习情绪评估",
                    description="依据学生自述、交互文本和行为线索判断当前情绪是否适合继续学习，并说明证据与不确定性",
                    task_type="reasoning",
                    requirement=CapabilityRequirement(reasoning=.86, interpretation=.94, domain_knowledge=.68, human_judgement=.58),
                    dependencies=["validate-evidence"],
                    input_contract=["learner_message", "emotion_observations[]", "recent_learning_context"],
                    output_contract=["emotion_needs_support", "emotional_distress_score", "emotion_evidence[]", "assessment_confidence"],
                    required_skills=["emotion_signal_interpretation", "supportive_language", "uncertainty_reporting"],
                    acceptance_criteria=["给出明确的布尔路由信号", "结论引用输入证据", "不进行临床诊断或标签化"],
                    runtime_hints={"model_family": "llm", "output_format": "json"},
                    execution_mode=ExecutionMode.PARALLEL,
                    risk=.5,
                ),
                Subtask(
                    id="emotion-router",
                    name="情绪路线判断",
                    description="综合情绪评估与知识诊断结果，确定进入情绪支持分支还是学习练习分支",
                    task_type="evaluation",
                    requirement=CapabilityRequirement(data_processing=.82, interpretation=.72),
                    dependencies=["estimate-mastery", "assess-emotion"],
                    input_contract=["emotion_needs_support", "emotional_distress_score", "mastery_by_knowledge_point{}"],
                    output_contract=["emotion_needs_support", "selected_route", "mastery_by_knowledge_point{}", "target_mastery"],
                    required_skills=["deterministic_branch_routing", "evidence_passthrough"],
                    acceptance_criteria=["只选择一个互斥分支", "保留后续学习诊断所需字段"],
                    runtime_hints={"operation": "route_emotion_state"},
                    risk=.2,
                ),
                Subtask(
                    id="emotion-support",
                    name="情绪安抚与学习支持",
                    description="在学生情绪状态不适合继续练习时，先提供共情回应、低负担支持和可选择的下一步，不强行布置练习",
                    task_type="generation",
                    requirement=CapabilityRequirement(reasoning=.84, generation=.9, interpretation=.92, human_judgement=.72),
                    dependencies=["emotion-router"],
                    input_contract=["emotion_evidence[]", "emotional_distress_score", "learner_message"],
                    output_contract=["support_message", "low_pressure_options[]", "escalation_recommendation", "pause_learning"],
                    required_skills=["empathetic_response", "psychological_safety", "non_clinical_support"],
                    acceptance_criteria=["先回应情绪再讨论任务", "不作临床诊断", "高风险信号建议寻求可信任的人或专业支持"],
                    runtime_hints={"model_family": "llm", "output_format": "json"},
                    execution_mode=ExecutionMode.CONDITIONAL,
                    entry_condition="emotion_needs_support == true",
                    risk=.55,
                ),
                Subtask(
                    id="emotion-support-confirm",
                    name="学习者确认情绪状态",
                    description="由学习者确认支持是否有帮助、是否希望暂停，以及何时愿意重新进入学习任务",
                    task_type="human_action",
                    requirement=CapabilityRequirement(human_judgement=.92, domain_knowledge=.25),
                    dependencies=["emotion-support"],
                    input_contract=["support_message", "low_pressure_options[]"],
                    output_contract=["support_helpful", "pause_learning", "ready_to_resume", "learner_note"],
                    required_skills=["learner_self_report"],
                    acceptance_criteria=["决定权由学习者保留", "允许暂停而不自动进入练习分支"],
                    runtime_hints={"form": "emotion_support_feedback"},
                    risk=.15,
                ),
            ])
        subtasks.extend([
            Subtask(
                id="rank-weak-points",
                name="薄弱知识点识别与排序",
                description="依据掌握度、目标差距和证据量识别薄弱知识点并按干预优先级排序",
                task_type="prediction",
                requirement=CapabilityRequirement(prediction=.84, interpretation=.84, data_processing=.72),
                dependencies=["emotion-router" if emotion_aware else "estimate-mastery"],
                input_contract=["mastery_by_knowledge_point{}", "target_mastery", "confidence"],
                output_contract=["weak_points[]", "weak_points[].priority", "weak_points_exist"],
                required_skills=["knowledge_gap_ranking", "evidence_weighting"],
                acceptance_criteria=["排序同时考虑掌握差距与证据置信度", "无薄弱点时返回明确的false状态"],
                runtime_hints={"algorithm": "weighted_gap_ranking"},
                execution_mode=ExecutionMode.CONDITIONAL if emotion_aware else ExecutionMode.SEQUENTIAL,
                entry_condition="emotion_needs_support == false" if emotion_aware else None,
                risk=.3,
            ),
            Subtask(
                id="design-practice",
                name="练习策略设计",
                description="为每个薄弱知识点确定练习目标、难度、题型、数量和递进顺序",
                task_type="reasoning",
                requirement=CapabilityRequirement(reasoning=.92, interpretation=.86, domain_knowledge=.82),
                dependencies=["rank-weak-points"],
                input_contract=["weak_points[]", "mastery_by_knowledge_point{}", "learner_context"],
                output_contract=["practice_blueprint[]", "practice_blueprint[].objective", "practice_blueprint[].difficulty"],
                required_skills=["instructional_design", "difficulty_adaptation", "knowledge_alignment"],
                acceptance_criteria=["每个薄弱点至少对应一个目标", "难度与当前掌握度匹配"],
                runtime_hints={"model_family": "llm", "output_format": "json"},
                execution_mode=ExecutionMode.CONDITIONAL,
                entry_condition="weak_points_exist == true",
                risk=.3,
            ),
            Subtask(
                id="generate-practice",
                name="薄弱点练习生成",
                description="严格按照练习策略生成题干、答案、解析、知识点和难度元数据",
                task_type="generation",
                requirement=CapabilityRequirement(reasoning=.82, generation=.97, interpretation=.86, domain_knowledge=.82),
                dependencies=["design-practice"],
                input_contract=["practice_blueprint[]", "learner_context"],
                output_contract=["exercises[]", "exercises[].question", "exercises[].answer", "exercises[].explanation", "exercises[].knowledge_point"],
                required_skills=["diagnostic_item_generation", "worked_explanation", "structured_json_output"],
                acceptance_criteria=["题目与目标知识点一致", "答案与解析自洽", "输出满足结构契约"],
                runtime_hints={"model_family": "llm", "output_format": "json"},
                risk=.35,
            ),
            Subtask(
                id="validate-practice",
                name="练习质量检查",
                description="检查练习结构完整性、答案存在性、知识点覆盖和明显重复项",
                task_type="evaluation",
                requirement=CapabilityRequirement(data_processing=.9, interpretation=.66),
                dependencies=["generate-practice"],
                input_contract=["exercises[]", "practice_blueprint[]"],
                output_contract=["validated_exercises[]", "quality.valid", "quality.issues[]"],
                required_skills=["exercise_schema_check", "coverage_check", "duplicate_detection"],
                acceptance_criteria=["所有练习包含答案、解析和知识点", "覆盖全部练习目标"],
                runtime_hints={"operation": "validate_exercise_set"},
                risk=.2,
            ),
        ])
        last_id = "validate-practice"
        if human_practice:
            subtasks.append(Subtask(
                id="human-practice",
                name="学习者完成练习",
                description="学习者查看练习并提交每题答案、解题过程、作答时长和主观困难",
                task_type="human_action",
                requirement=CapabilityRequirement(domain_knowledge=.58, human_judgement=.65),
                dependencies=[last_id],
                input_contract=["validated_exercises[]"],
                output_contract=["learner_responses[]", "learner_responses[].answer", "learner_responses[].duration_seconds", "learner_responses[].difficulty_feedback"],
                required_skills=["learner_response_collection"],
                acceptance_criteria=["每道题均有答案或明确跳过状态", "保留题目ID以便自动评分"],
                runtime_hints={"form": "exercise_response"},
                risk=.1,
            ))
            last_id = "human-practice"
        if evaluation:
            subtasks.extend([
                Subtask(
                    id="score-practice",
                    name="练习作答评分",
                    description="依据标准答案对学习者作答进行逐题评分并聚合到知识点",
                    task_type="evaluation",
                    requirement=CapabilityRequirement(data_processing=.94, prediction=.62, interpretation=.58),
                    dependencies=[last_id],
                    input_contract=["validated_exercises[]", "learner_responses[]"],
                    output_contract=["item_scores[]", "score_by_knowledge_point{}", "overall_score"],
                    required_skills=["deterministic_scoring", "knowledge_point_aggregation"],
                    acceptance_criteria=["每题分数可追溯到题目ID", "总体分数与知识点分数一致"],
                    runtime_hints={"operation": "score_learning_responses"},
                    risk=.2,
                ),
                Subtask(
                    id="update-mastery",
                    name="学习效果与掌握度更新",
                    description="融合本轮评分与历史估计，更新知识点掌握度并判断是否达到学习目标",
                    task_type="prediction",
                    requirement=CapabilityRequirement(prediction=.9, data_processing=.82, interpretation=.78),
                    dependencies=["score-practice"],
                    input_contract=["score_by_knowledge_point{}", "mastery_by_knowledge_point{}", "target_mastery"],
                    output_contract=["updated_mastery{}", "goal_reached", "remaining_weak_points[]", "improvement"],
                    required_skills=["online_mastery_update", "goal_threshold_decision"],
                    acceptance_criteria=["更新结果在0到1之间", "达标判断使用显式阈值"],
                    runtime_hints={"algorithm": "weighted_mastery_update"},
                    risk=.3,
                ),
                Subtask(
                    id="analyse-feedback",
                    name="下一轮反馈与调整",
                    description="解释本轮错误模式并为尚未掌握的知识点形成下一轮练习调整建议",
                    task_type="reasoning",
                    requirement=CapabilityRequirement(reasoning=.92, generation=.82, interpretation=.9, domain_knowledge=.74),
                    dependencies=["update-mastery"],
                    input_contract=["updated_mastery{}", "remaining_weak_points[]", "item_scores[]"],
                    output_contract=["error_patterns[]", "adjustment_plan[]", "needs_revision", "goal_reached"],
                    required_skills=["error_pattern_analysis", "adaptive_feedback", "revision_planning"],
                    acceptance_criteria=["每项调整均引用评分证据", "已达标时不再生成无意义练习"],
                    runtime_hints={"model_family": "llm", "output_format": "json"},
                    execution_mode=ExecutionMode.ITERATIVE if iterative else ExecutionMode.SEQUENTIAL,
                    iteration_policy=IterationPolicy(
                        enabled=iterative,
                        feedback_target_subtask_id="design-practice" if iterative else None,
                        condition="goal_reached == false",
                        max_iterations=3,
                    ),
                    risk=.3,
                ),
            ])
            last_id = "analyse-feedback"
        if summary_generation:
            subtasks.append(Subtask(
                id="summary-report",
                name="学习总结报告",
                description="汇总初始诊断、各轮练习、掌握度变化、达标状态与后续建议",
                task_type="generation",
                requirement=CapabilityRequirement(reasoning=.78, generation=.94, interpretation=.9),
                dependencies=[last_id],
                input_contract=["mastery_by_knowledge_point{}", "updated_mastery{}", "item_scores[]", "goal_reached"],
                output_contract=["summary", "mastery_changes[]", "evidence[]", "next_steps[]"],
                required_skills=["evidence_synthesis", "learning_progress_reporting"],
                acceptance_criteria=["结论可追溯到诊断与练习证据", "明确区分事实、推断和建议"],
                runtime_hints={"model_family": "llm", "output_format": "json"},
                execution_mode=ExecutionMode.CONDITIONAL if evaluation else ExecutionMode.SEQUENTIAL,
                entry_condition="goal_reached == true or max_iterations_reached == true" if evaluation else None,
                risk=.25,
            ))
        breadth = sum(1 for item in subtasks for value in item.requirement.model_dump().values() if value >= .7)
        complexity = min(.98, .18 + len(subtasks) * .055 + breadth * .018)
        return TaskGraph(
            task_id=task_id,
            goal=prompt,
            complexity=round(complexity, 3),
            subtasks=subtasks,
            planning_mode="rule_atomic_workflow",
        )

    async def understand_async(self, prompt: str, model_config: dict | None = None) -> TaskGraph:
        """Prefer model-based structured planning and fall back to deterministic rules."""
        if not model_config or not model_config.get("api_key"):
            return self.understand(prompt)
        try:
            llm = ChatOpenAI(
                api_key=model_config["api_key"],
                base_url=model_config.get("base_url") or None,
                model=model_config.get("model") or "gpt-4.1-mini",
                temperature=0,
                timeout=45,
            ).with_structured_output(GeneratedBreakdown)
            result = await llm.ainvoke(
                "你是 Adaptive-AGES 任务规划器。把用户需求拆成 2 到 16 个原子、可执行子任务；"
                "同一种主体可以承担多个职责不同的节点，不得套用每种主体一个节点的模板。"
                "每个子任务选择清晰的任务类型：data_processing、prediction、reasoning、generation、"
                "evaluation、human_action、human_review 或 tool，并提供输入输出契约、技能、验收标准和运行提示。"
                "能力需求字段取 0 到 1；依赖必须只指向列表中更早出现的子任务 id。"
                "只有涉及高风险、价值判断、责任确认或用户明确要求时才生成人类复核节点。\n\n"
                f"用户需求：{prompt}"
            )
            seen: set[str] = set()
            cleaned: list[Subtask] = []
            for index, item in enumerate(result.subtasks[:16]):
                item_id = item.id.strip() or f"step-{index + 1}"
                if item_id in seen:
                    item_id = f"{item_id}-{index + 1}"
                dependencies = [dep for dep in item.dependencies if dep in seen]
                cleaned.append(Subtask(**item.model_dump(exclude={"id", "dependencies"}), id=item_id, dependencies=dependencies))
                seen.add(item_id)
            if not cleaned:
                return self.understand(prompt)
            breadth = sum(1 for s in cleaned for value in s.requirement.model_dump().values() if value >= .7)
            complexity = min(.98, .18 + len(cleaned) * .14 + breadth * .035)
            task_id = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
            return TaskGraph(task_id=task_id, goal=prompt, complexity=round(complexity, 3), subtasks=cleaned, planning_mode="llm")
        except Exception:  # noqa: BLE001 - provider failures must fall back to local planning
            return self.understand(prompt)
