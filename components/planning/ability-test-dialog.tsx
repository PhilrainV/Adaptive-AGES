"use client";

import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  GitBranch,
  LoaderCircle,
  SearchCheck,
  Sparkles,
  X,
} from "lucide-react";

export interface AbilityQuestion {
  id: string;
  dimension: string;
  prompt: string;
  options: string[];
}

export interface PlanningAgentTrace {
  agent: string;
  status: string;
  summary?: string;
  mode?: string;
}

const dimensionLabels: Record<string, string> = {
  programming: "编程与实现",
  ai_literacy: "AI 素养",
  domain_knowledge: "领域知识",
  workflow_design: "流程设计",
  judgement: "专业判断",
};

const agents = [
  { id: "problem_analysis_agent", label: "问题解析", icon: SearchCheck },
  { id: "test_generation_agent", label: "动态出题", icon: Sparkles },
  { id: "ability_diagnosis_agent", label: "能力诊断", icon: BrainCircuit },
  { id: "capability_planning_agent", label: "任务规划", icon: GitBranch },
];

export function AbilityTestDialog({
  questions,
  trace,
  submitting,
  onCancel,
  onSubmit,
}: {
  questions: AbilityQuestion[];
  trace: PlanningAgentTrace[];
  submitting: boolean;
  onCancel: () => void;
  onSubmit: (answers: Record<string, number>) => void;
}) {
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number>>({});

  if (!questions.length) return null;
  const question = questions[index];
  const selected = answers[question.id];
  const answered = Object.keys(answers).length;

  return <div className="assessment-overlay" role="dialog" aria-modal="true" aria-label="任务自适应能力测试">
    <div className="assessment-dialog">
      <div className="assessment-dialog-header">
        <div><p className="eyebrow">Assessment before planning</p><h3>先测试本次任务所需能力</h3><p>测试结果仅用于校准 Human Agent，再由规划 Agent 决定人、LLM、ML 与工具如何协作。</p></div>
        <button className="dialog-close" onClick={onCancel} disabled={submitting} aria-label="取消测试"><X size={17}/></button>
      </div>

      <div className="planning-agent-strip">
        {agents.map((agent, agentIndex) => {
          const item = trace.find(entry => entry.agent === agent.id);
          const complete = item?.status === "completed" || (submitting && agentIndex < 2);
          const active = submitting && agentIndex === 2;
          const Icon = agent.icon;
          return <div className={`planning-agent ${complete ? "complete" : ""} ${active ? "active" : ""}`} key={agent.id}>
            <span>{complete ? <CheckCircle2 size={15}/> : active ? <LoaderCircle className="spin" size={15}/> : <Icon size={15}/>}</span>
            <div><strong>{agentIndex + 1}. {agent.label} Agent</strong><small>{item?.summary || (active ? "正在根据答题证据诊断…" : "等待上一阶段")}</small></div>
          </div>;
        })}
      </div>

      <div className="assessment-progress">
        <div><span>{dimensionLabels[question.dimension] || question.dimension}</span><strong>第 {index + 1} / {questions.length} 题</strong></div>
        <div className="assessment-progress-track"><span style={{width:`${((index + 1) / questions.length) * 100}%`}}/></div>
      </div>

      <div className="assessment-question-one">
        <h4>{question.prompt}</h4>
        <div className="assessment-options-one">
          {question.options.map((option, optionIndex) => <button
            type="button"
            className={selected === optionIndex ? "selected" : ""}
            key={`${question.id}-${optionIndex}`}
            onClick={() => setAnswers(current => ({...current, [question.id]: optionIndex}))}
            disabled={submitting}
          ><span>{String.fromCharCode(65 + optionIndex)}</span>{option}</button>)}
        </div>
      </div>

      <div className="assessment-dialog-footer">
        <span>已完成 {answered}/{questions.length}</span>
        <div>
          <button className="ghost-button" onClick={() => setIndex(value => value - 1)} disabled={index === 0 || submitting}><ArrowLeft size={14}/>上一题</button>
          {index < questions.length - 1
            ? <button className="primary-button lime" onClick={() => setIndex(value => value + 1)} disabled={selected === undefined || submitting}>下一题<ArrowRight size={14}/></button>
            : <button className="primary-button" onClick={() => onSubmit(answers)} disabled={answered !== questions.length || submitting}>{submitting ? <LoaderCircle className="spin" size={14}/> : <GitBranch size={14}/>}诊断并生成规划</button>}
        </div>
      </div>
    </div>
  </div>;
}
