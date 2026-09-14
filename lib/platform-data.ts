import type { Edge, Node } from "@xyflow/react";

export type ViewKey = "dashboard" | "studio" | "capabilities" | "profile" | "settings";
export type AgentKind = "LLM" | "ML" | "Human" | "Tool" | "Knowledge" | "Adaptive";
export type NodeState = "idle" | "running" | "done" | "waiting";

export interface WorkflowNodeData extends Record<string, unknown> {
  label: string;
  kind: AgentKind;
  subtitle: string;
  score: number;
  reason: string;
  capabilities: Record<string, number>;
  state?: NodeState;
  subjectId?: string;
  subtaskId?: string;
  config?: Record<string, unknown>;
}

export const initialNodes: Node<WorkflowNodeData>[] = [
  { id: "input", type: "workflow", position: { x: 30, y: 190 }, data: { label: "学习数据接入", kind: "Tool", subtitle: "CSV / API Connector", score: 1, reason: "原始数据需要统一完成字段校验、缺失值处理和特征装配。", capabilities: { 数据处理: .93, 稳定性: .96 }, state: "done" } },
  { id: "adaptive", type: "workflow", position: { x: 265, y: 190 }, data: { label: "主体自适应决策", kind: "Adaptive", subtitle: "Capability Router", score: .92, reason: "当前任务同时需要高精度预测、教学语义解释与最终责任判断，因此采用 ML → LLM → Human 的异构协同路径。", capabilities: { 预测: .88, 推理: .90, 人类判断: .76 }, state: "done" } },
  { id: "ml", type: "workflow", position: { x: 520, y: 70 }, data: { label: "学业风险预测", kind: "ML", subtitle: "XGBoost · v2.4", score: .96, reason: "该子任务以结构化小样本预测为主；XGBoost 在历史评测中的 AUC、稳定性和可解释性综合得分最高。", capabilities: { 预测: .96, 小样本: .88, 可解释: .84 }, state: "idle" } },
  { id: "llm", type: "workflow", position: { x: 520, y: 190 }, data: { label: "教学建议生成", kind: "LLM", subtitle: "GPT-4.1 Agent", score: .94, reason: "需要将风险特征转化为连贯、个性化且符合教学约束的自然语言建议，因此选择高推理与解释能力的 LLM。", capabilities: { 推理: .95, 解释: .94, 生成: .93 }, state: "idle" } },
  { id: "human", type: "workflow", position: { x: 520, y: 310 }, data: { label: "教师专业复核", kind: "Human", subtitle: "学科教师 · Level 4", score: .89, reason: "建议涉及真实教学决策与学生权益，需要教师结合课堂情境作最终校准并承担决策责任。", capabilities: { 领域知识: .95, 判断: .92, 责任: .98 }, state: "waiting" } },
  { id: "output", type: "workflow", position: { x: 775, y: 190 }, data: { label: "个性化方案", kind: "Knowledge", subtitle: "Structured Output", score: .91, reason: "汇总预测结果、解释与教师修订，形成可追溯的学生支持方案。", capabilities: { 完整性: .94, 可追溯: .91 }, state: "idle" } },
];

export const initialEdges: Edge[] = [
  { id: "e1", source: "input", target: "adaptive", animated: true },
  { id: "e2", source: "adaptive", target: "ml", label: "预测", animated: true },
  { id: "e3", source: "adaptive", target: "llm", label: "解释", animated: true },
  { id: "e4", source: "adaptive", target: "human", label: "判断", animated: true },
  { id: "e5", source: "ml", target: "output" },
  { id: "e6", source: "llm", target: "output" },
  { id: "e7", source: "human", target: "output" },
];

export const capabilityAgents = [
  { name: "GPT-4.1 Reasoner", kind: "LLM Agent", color: "#5d7cff", values: [95, 91, 94, 86] },
  { name: "XGBoost Risk", kind: "ML Model", color: "#f3a950", values: [72, 96, 84, 94] },
  { name: "教师专家组", kind: "Human", color: "#9a76e8", values: [90, 73, 96, 88] },
  { name: "Data Processor", kind: "Tool", color: "#7e9a88", values: [61, 89, 78, 98] },
];

export const radarData = [
  { dimension: "推理", LLM: 95, ML: 72, Human: 90 },
  { dimension: "预测", LLM: 78, ML: 96, Human: 73 },
  { dimension: "解释", LLM: 94, ML: 84, Human: 96 },
  { dimension: "稳定", LLM: 86, ML: 94, Human: 88 },
  { dimension: "领域", LLM: 81, ML: 76, Human: 97 },
  { dimension: "成本", LLM: 72, ML: 91, Human: 54 },
];
