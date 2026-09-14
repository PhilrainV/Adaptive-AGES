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

export interface AbilityDiagnosis {
  overall: number;
  confidence: number;
  capability: Record<string, number>;
  planning_capability: Record<string, number>;
  weakest_dimensions: string[];
  answered?: number;
  total?: number;
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
  diagnosis,
  diagnosing,
  planning,
  onCancel,
  onSubmitAnswers,
  onStartPlanning,
}: {
  questions: AbilityQuestion[];
  trace: PlanningAgentTrace[];
  diagnosis: AbilityDiagnosis | null;
  diagnosing: boolean;
  planning: boolean;
  onCancel: () => void;
  onSubmitAnswers: (answers: Record<string, number>) => void;
  onStartPlanning: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, number>>({});

  if (!questions.length) return null;
  const question = questions[index];
  const selected = answers[question.id];
  const answered = Object.keys(answers).length;
  const busy = diagnosing || planning;

  return <div className="assessment-overlay" role="dialog" aria-modal="true" aria-label="任务自适应能力测试">
    <div className="assessment-dialog">
      <div className="assessment-dialog-header">
        <div>
          <p className="eyebrow">{diagnosis ? "Diagnosis completed" : "Assessment before planning"}</p>
          <h3>{diagnosis ? "能力诊断已完成" : "先测试本次任务所需能力"}</h3>
          <p>{diagnosis
            ? "当前只生成了诊断结果，工作流尚未创建。确认结果后，再单独启动规划 Agent。"
            : "测试结果仅用于校准 Human Agent；提交后先查看诊断，不会立即生成工作流。"}</p>
        </div>
        <button className="dialog-close" onClick={onCancel} disabled={busy} aria-label="关闭"><X size={17}/></button>
      </div>

      <div className="planning-agent-strip">
        {agents.map((agent, agentIndex) => {
          const item = trace.find(entry => entry.agent === agent.id);
          const complete = item?.status === "completed";
          const active = (diagnosing && agentIndex === 2) || (planning && agentIndex === 3);
          const Icon = agent.icon;
          const fallback = active
            ? agentIndex === 2 ? "正在根据答题证据诊断…" : "正在根据任务与诊断结果规划…"
            : agentIndex === 3 && diagnosis ? "等待你确认后启动" : "等待上一阶段";
          return <div className={`planning-agent ${complete ? "complete" : ""} ${active ? "active" : ""}`} key={agent.id}>
            <span>{complete ? <CheckCircle2 size={15}/> : active ? <LoaderCircle className="spin" size={15}/> : <Icon size={15}/>}</span>
            <div><strong>{agentIndex + 1}. {agent.label} Agent</strong><small>{item?.summary || fallback}</small></div>
          </div>;
        })}
      </div>

      {diagnosis ? <div className="diagnosis-stage">
        <div className="diagnosis-summary-grid">
          <div className="diagnosis-summary-card">
            <span>综合能力</span>
            <strong>{Math.round(diagnosis.overall * 100)}%</strong>
            <small>用于本次任务的人类能力基线</small>
          </div>
          <div className="diagnosis-summary-card">
            <span>诊断置信度</span>
            <strong>{Math.round(diagnosis.confidence * 100)}%</strong>
            <small>{diagnosis.answered ?? questions.length}/{diagnosis.total ?? questions.length} 道证据已纳入</small>
          </div>
        </div>

        <div className="diagnosis-panel">
          <div className="diagnosis-panel-heading">
            <div><strong>能力维度</strong><span>规划 Agent 将使用这些结果校准 Human 节点</span></div>
            <span className="diagnosis-stage-pill">阶段 3 已保存</span>
          </div>
          <div className="diagnosis-bars">
            {Object.entries(diagnosis.capability).map(([name, value]) => <div className="diagnosis-bar-row" key={name}>
              <div><span>{dimensionLabels[name] || name}</span><strong>{Math.round(value * 100)}%</strong></div>
              <div className="cap-track"><div className="cap-fill" style={{width:`${value * 100}%`}}/></div>
            </div>)}
          </div>
          {!!diagnosis.weakest_dimensions.length && <div className="diagnosis-weaknesses">
            <span>建议重点由模型或工具辅助：</span>
            {diagnosis.weakest_dimensions.map(name => <strong key={name}>{dimensionLabels[name] || name}</strong>)}
          </div>}
        </div>

        <div className="diagnosis-boundary-note">
          <GitBranch size={17}/>
          <div><strong>规划尚未开始</strong><span>点击下方按钮后，系统才会把任务图和这份诊断交给任务规划 Agent，并创建工作流。</span></div>
        </div>
      </div> : <>
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
              disabled={busy}
            ><span>{String.fromCharCode(65 + optionIndex)}</span>{option}</button>)}
          </div>
        </div>
      </>}

      <div className="assessment-dialog-footer">
        <span>{diagnosis ? "诊断结果已保存，可稍后从总览继续规划" : `已完成 ${answered}/${questions.length}`}</span>
        <div>
          {diagnosis
            ? <>
              <button className="ghost-button" onClick={onCancel} disabled={busy}>暂不规划</button>
              <button className="primary-button" onClick={onStartPlanning} disabled={busy}>
                {planning ? <LoaderCircle className="spin" size={14}/> : <GitBranch size={14}/>}
                {planning ? "规划中…" : "基于诊断结果开始规划"}
              </button>
            </>
            : <>
              <button className="ghost-button" onClick={() => setIndex(value => value - 1)} disabled={index === 0 || busy}><ArrowLeft size={14}/>上一题</button>
              {index < questions.length - 1
                ? <button className="primary-button lime" onClick={() => setIndex(value => value + 1)} disabled={selected === undefined || busy}>下一题<ArrowRight size={14}/></button>
                : <button className="primary-button" onClick={() => onSubmitAnswers(answers)} disabled={answered !== questions.length || busy}>
                  {diagnosing ? <LoaderCircle className="spin" size={14}/> : <BrainCircuit size={14}/>}
                  {diagnosing ? "诊断中…" : "提交测试并查看诊断"}
                </button>}
            </>}
        </div>
      </div>
    </div>
  </div>;
}
