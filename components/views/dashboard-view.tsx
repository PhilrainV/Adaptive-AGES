"use client";

import { ArrowRight, Bot, CheckCircle2, GitBranch, Play, Sparkles, Users } from "lucide-react";

const metrics = [
  { label: "可用智能主体", value: "18", delta: "+3 本月", icon: Bot, bg: "#edf2ff", color: "#5d7cff" },
  { label: "自适应工作流", value: "12", delta: "+21%", icon: GitBranch, bg: "#f3fae7", color: "#719c29" },
  { label: "本月执行", value: "1,284", delta: "+16.8%", icon: Play, bg: "#fff3e5", color: "#d7872d" },
  { label: "协同成功率", value: "94.2%", delta: "+2.4%", icon: CheckCircle2, bg: "#f2ecff", color: "#8b62db" },
];

export function DashboardView({ onNavigate }: { onNavigate: () => void }) {
  return <>
    <div className="section-heading"><div><p className="eyebrow">Overview</p><h2>下午好，魏老师</h2><p>3 个工作流正在运行，1 个节点等待教师确认。</p></div><button className="primary-button" onClick={onNavigate}><Sparkles size={15}/>创建自适应工作流</button></div>
    <div className="metric-grid">{metrics.map(({ label, value, delta, icon: Icon, bg, color }) => <div className="metric-card" key={label}><div className="metric-top"><span className="metric-icon" style={{background:bg,color}}><Icon size={17}/></span><span className="metric-delta">{delta}</span></div><div className="metric-value">{value}</div><div className="metric-label">{label}</div></div>)}</div>
    <div className="dashboard-grid">
      <section className="panel"><div className="panel-header"><div><h3>最近任务</h3><p>异构主体协同执行情况</p></div><button className="ghost-button" onClick={onNavigate}>查看全部 <ArrowRight size={13}/></button></div><div className="table-wrap"><table className="data-table"><thead><tr><th>任务</th><th>主体组合</th><th>状态</th><th>匹配分</th></tr></thead><tbody>
        <tr><td><div className="task-name">学生学业风险分析</div><div className="task-sub">8 分钟前更新</div></td><td><div className="agent-stack"><span className="agent-dot" style={{background:"#f3a950"}}>ML</span><span className="agent-dot" style={{background:"#5d7cff"}}>AI</span><span className="agent-dot" style={{background:"#9a76e8"}}>人</span></div></td><td><span className="status-pill live">运行中</span></td><td>92%</td></tr>
        <tr><td><div className="task-name">开放题自动评分与复核</div><div className="task-sub">昨天 16:20</div></td><td><div className="agent-stack"><span className="agent-dot" style={{background:"#5d7cff"}}>AI</span><span className="agent-dot" style={{background:"#9a76e8"}}>人</span></div></td><td><span className="status-pill review">待复核</span></td><td>89%</td></tr>
        <tr><td><div className="task-name">课程知识图谱更新</div><div className="task-sub">9 月 12 日</div></td><td><div className="agent-stack"><span className="agent-dot" style={{background:"#5d7cff"}}>AI</span><span className="agent-dot" style={{background:"#7e9a88"}}>工具</span></div></td><td><span className="status-pill draft">已完成</span></td><td>95%</td></tr>
      </tbody></table></div></section>
      <section className="panel"><div className="panel-header"><div><h3>实时动态</h3><p>来自规划器与执行引擎</p></div><Users size={16} color="#718078"/></div><div className="activity-list"><div className="activity-item"><span className="activity-marker accent"/><div><div className="activity-title">教师完成教学建议复核</div><div className="activity-meta">纠正内容已用于更新能力画像</div></div><span className="activity-time">2m</span></div><div className="activity-item"><span className="activity-marker accent"/><div><div className="activity-title">规划器切换至 XGBoost</div><div className="activity-meta">小样本预测匹配度提高 7%</div></div><span className="activity-time">8m</span></div><div className="activity-item"><span className="activity-marker"/><div><div className="activity-title">GPT-4.1 Agent 能力校准</div><div className="activity-meta">解释维度 0.91 → 0.94</div></div><span className="activity-time">1h</span></div></div></section>
    </div>
  </>;
}
