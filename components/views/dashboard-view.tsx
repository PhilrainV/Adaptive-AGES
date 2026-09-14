"use client";

import { useCallback, useEffect, useState } from "react";
import { ArrowRight, Bot, CheckCircle2, GitBranch, LoaderCircle, Play, Sparkles, Trash2, Users } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface DashboardData {
  display_name: string;
  metrics: {
    agents: number;
    workflows: number;
    month_executions: number;
    success_rate: number;
    running: number;
    waiting_human: number;
    failed: number;
  };
  recent_tasks: Array<{
    id: string;
    workflow_id: string | null;
    title: string;
    agents: string[];
    status: string;
    match_score: number;
    updated_at: string;
  }>;
  activities: Array<{ id:string; title:string; status:string; created_at:string }>;
}

const statusLabel: Record<string,string> = {
  ready:"就绪", draft:"草稿", queued:"排队中", running:"运行中", completed:"已完成",
  failed:"失败", waiting_for_human:"待人工处理", resumable:"可继续", assessing:"待能力测试",
};
const agentLabel: Record<string,string> = {llm:"LLM",ml:"ML",human:"人",tool:"工具"};
const agentColor: Record<string,string> = {llm:"#5d7cff",ml:"#f3a950",human:"#9a76e8",tool:"#7e9a88"};
const statusClass = (status:string) => status === "running" || status === "queued" ? "live" : status === "waiting_for_human" || status === "failed" ? "review" : "draft";
const formatTime = (value:string) => new Intl.DateTimeFormat("zh-CN",{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}).format(new Date(value));

export function DashboardView({
  onCreate,
  onOpenTask,
  notify,
}: {
  onCreate: () => void;
  onOpenTask: (taskId:string) => void;
  notify: (message:string) => void;
}) {
  const [data,setData] = useState<DashboardData | null>(null);
  const [error,setError] = useState("");
  const [deletingId,setDeletingId] = useState("");
  const load = useCallback(() => {
    return apiFetch<DashboardData>("/dashboard").then(setData).catch(reason => setError(reason instanceof Error ? reason.message : "总览加载失败"));
  }, []);
  useEffect(() => {
    void load();
  }, [load]);

  const deleteTask = async (taskId:string, title:string) => {
    if (!window.confirm(`确定删除任务“${title}”吗？相关工作流和执行记录也会一并删除。`)) return;
    setDeletingId(taskId);
    try {
      await apiFetch<{deleted:boolean}>(`/tasks/${taskId}`, {method:"DELETE"});
      await load();
      notify("任务及其工作流已删除");
    } catch (reason) { notify(reason instanceof Error ? reason.message : "删除任务失败"); }
    finally { setDeletingId(""); }
  };

  if (!data) return <div className="panel config-placeholder">{error || <><LoaderCircle className="spin" size={17}/>正在读取真实工作区数据…</>}</div>;
  const metrics = [
    { label:"已注册智能主体", value:String(data.metrics.agents), delta:"当前用户", icon:Bot, bg:"#edf2ff", color:"#5d7cff" },
    { label:"自适应工作流", value:String(data.metrics.workflows), delta:`${data.metrics.running} 个运行中`, icon:GitBranch, bg:"#f3fae7", color:"#719c29" },
    { label:"本月执行", value:String(data.metrics.month_executions), delta:`${data.metrics.waiting_human} 个待人工`, icon:Play, bg:"#fff3e5", color:"#d7872d" },
    { label:"实际执行成功率", value:`${(data.metrics.success_rate*100).toFixed(1)}%`, delta:`${data.metrics.failed} 个失败`, icon:CheckCircle2, bg:"#f2ecff", color:"#8b62db" },
  ];

  return <>
    <div className="section-heading"><div><p className="eyebrow">Overview</p><h2>你好，{data.display_name}</h2><p>{data.metrics.running} 个工作流正在运行，{data.metrics.waiting_human} 个执行等待人工处理。</p></div><button className="primary-button" onClick={onCreate}><Sparkles size={15}/>创建自适应工作流</button></div>
    <div className="metric-grid">{metrics.map(({label,value,delta,icon:Icon,bg,color}) => <div className="metric-card" key={label}><div className="metric-top"><span className="metric-icon" style={{background:bg,color}}><Icon size={17}/></span><span className="metric-delta">{delta}</span></div><div className="metric-value">{value}</div><div className="metric-label">{label}</div></div>)}</div>
    <div className="dashboard-grid">
      <section className="panel"><div className="panel-header"><div><h3>最近任务</h3><p>点击任务进入对应工作流；可在最右侧删除</p></div><button className="ghost-button" onClick={onCreate}>新建任务 <ArrowRight size={13}/></button></div><div className="table-wrap"><table className="data-table"><thead><tr><th>任务</th><th>主体组合</th><th>状态</th><th>匹配分</th><th aria-label="操作"/></tr></thead><tbody>
        {data.recent_tasks.map(item => <tr className="task-table-row" key={item.id} tabIndex={0} onClick={() => onOpenTask(item.id)} onKeyDown={event => {if (event.key === "Enter") onOpenTask(item.id)}}><td><div className="task-name">{item.title}</div><div className="task-sub">{formatTime(item.updated_at)} 更新</div></td><td><div className="agent-stack">{item.agents.map(kind => <span className="agent-dot" key={kind} style={{background:agentColor[kind] || "#7e9a88"}}>{agentLabel[kind] || kind}</span>)}</div></td><td><span className={`status-pill ${statusClass(item.status)}`}>{statusLabel[item.status] || item.status}</span></td><td>{Math.round(item.match_score*100)}%</td><td><button className="task-delete-button" title="删除任务" aria-label={`删除任务 ${item.title}`} disabled={deletingId === item.id} onClick={event => {event.stopPropagation();void deleteTask(item.id,item.title)}}>{deletingId === item.id ? <LoaderCircle className="spin" size={14}/> : <Trash2 size={14}/>}</button></td></tr>)}
        {!data.recent_tasks.length && <tr><td colSpan={5}><div className="config-placeholder">还没有真实工作流。点击“创建自适应工作流”开始规划。</div></td></tr>}
      </tbody></table></div></section>
      <section className="panel"><div className="panel-header"><div><h3>执行动态</h3><p>最近的真实执行事件</p></div><Users size={16} color="#718078"/></div><div className="activity-list">
        {data.activities.map(item => <div className="activity-item" key={item.id}><span className={`activity-marker ${item.status === "completed" ? "accent" : ""}`}/><div><div className="activity-title">{item.title}</div><div className="activity-meta">{statusLabel[item.status] || item.status}</div></div><span className="activity-time">{formatTime(item.created_at)}</span></div>)}
        {!data.activities.length && <div className="config-placeholder">暂无执行记录。运行工作流后这里会自动更新。</div>}
      </div></section>
    </div>
  </>;
}
