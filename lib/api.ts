"use client";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api/v1";
const SESSION_KEY = "adaptive-ages-session";

type Session = { email: string; password: string; token?: string };

function session(): Session {
  const saved = window.localStorage.getItem(SESSION_KEY);
  if (saved) return JSON.parse(saved) as Session;
  const id = crypto.randomUUID().slice(0, 12);
  const created = { email: `workspace-${id}@adaptive.local`, password: `Aa!${crypto.randomUUID()}` };
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(created));
  return created;
}

async function authenticate(current: Session): Promise<string> {
  const register = await fetch(`${API_BASE}/auth/register`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: current.email, password: current.password, display_name: "Local Workspace" }),
  });
  let response = register;
  if (register.status === 409) {
    response = await fetch(`${API_BASE}/auth/login`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: current.email, password: current.password }),
    });
  }
  if (!response.ok) throw new Error("无法创建本地工作区会话");
  const body = await response.json() as { access_token: string };
  current.token = body.access_token;
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(current));
  return current.token as string;
}

async function token(force = false): Promise<string> {
  const current = session();
  return !force && current.token ? current.token : authenticate(current);
}

export async function apiFetch<T>(path: string, init: RequestInit = {}, retry = true): Promise<T> {
  const accessToken = await token();
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}`, ...(init.headers || {}) },
  });
  if (response.status === 401 && retry) {
    await token(true);
    return apiFetch<T>(path, init, false);
  }
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { const body = await response.json() as {detail?:string}; message = body.detail || message; } catch {}
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function apiDownload(path: string): Promise<Blob> {
  const accessToken = await token();
  const response = await fetch(`${API_BASE}${path}`, { headers: { Authorization: `Bearer ${accessToken}` } });
  if (!response.ok) throw new Error(`导出失败 (${response.status})`);
  return response.blob();
}

export type SubjectType = "llm" | "ml" | "human" | "tool";

export interface WorkflowPlanNode {
  id: string;
  subtask_id: string;
  subject_id: string;
  subject_type: SubjectType;
  label: string;
  match_score: number;
  config: Record<string, unknown>;
  explainability: { reason?: string; similarity?: number; alternatives?: Array<Record<string, unknown>> };
}

export interface WorkflowPlan {
  id: string;
  task_id: string;
  nodes: WorkflowPlanNode[];
  edges: Array<{ source: string; target: string; condition?: string | null }>;
  decision_trace: Array<Record<string, unknown>>;
  estimated_cost: number;
  requires_human: boolean;
}

export const defaultCapabilitySpace = [
  { id:"gpt-agent", name:"General LLM Agent", subject_type:"llm", capability:{ reasoning:.96, generation:.96, interpretation:.92, domain_knowledge:.72, data_processing:.45 }, reliability:.9, cost:.32, latency:.25 },
  { id:"ml-python", name:"Python ML Runtime", subject_type:"ml", capability:{ prediction:.97, data_processing:.82, interpretation:.76, reasoning:.3 }, reliability:.94, cost:.08, latency:.08 },
  { id:"human-user", name:"Task Human Expert", subject_type:"human", capability:{ human_judgement:.96, domain_knowledge:.93, interpretation:.9, reasoning:.86 }, reliability:.9, cost:.72, latency:.8 },
  { id:"data-tool", name:"Data & API Tool", subject_type:"tool", capability:{ data_processing:.98, prediction:.24, interpretation:.34 }, reliability:.97, cost:.04, latency:.04 },
];
