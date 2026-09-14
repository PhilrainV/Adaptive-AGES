"use client";

import { useMemo, useState } from "react";
import { Bot, BrainCircuit, Database, GitBranch, LoaderCircle, Play, Search, SlidersHorizontal, Sparkles, UserRound, Wrench } from "lucide-react";
import { WorkflowCanvas } from "@/components/workflow/workflow-canvas";
import { initialNodes } from "@/lib/platform-data";

const components = [
  { name:"LLM Agent", desc:"推理、生成与解释", icon:Bot, bg:"#edf2ff", color:"#5d7cff" },
  { name:"ML Model", desc:"预测、分类与分析", icon:BrainCircuit, bg:"#fff3e5", color:"#d7872d" },
  { name:"Human Expert", desc:"判断、复核与反馈", icon:UserRound, bg:"#f2ecff", color:"#8b62db" },
  { name:"Adaptive Router", desc:"按能力动态分配", icon:GitBranch, bg:"#edfad7", color:"#5f8d1f" },
  { name:"Tool", desc:"API 与数据工具", icon:Wrench, bg:"#edf3ef", color:"#5f7c6b" },
  { name:"Knowledge", desc:"知识库与长期记忆", icon:Database, bg:"#e9f7f0", color:"#3a8a64" },
];

export function StudioView({ notify }: { notify:(m:string)=>void }) {
  const [task,setTask] = useState("分析学生学习数据，预测学业风险，并生成个性化教学建议，最后由教师复核");
  const [selectedId,setSelectedId] = useState("adaptive");
  const [planning,setPlanning] = useState(false);
  const [runToken,setRunToken] = useState(0);
  const selected = useMemo(() => initialNodes.find(n => n.id === selectedId)?.data ?? initialNodes[1].data, [selectedId]);
  const plan = () => { setPlanning(true); window.setTimeout(() => { setPlanning(false); setSelectedId("adaptive"); notify("已生成 6 节点异构协同工作流"); },850); };
  const run = () => { setRunToken(v => v+1); notify("工作流已进入执行队列"); };

  return <div className="studio-shell">
    <aside className="component-panel"><div className="studio-panel-title"><h3>智能主体</h3><SlidersHorizontal size={14}/></div><div className="component-search"><Search size={14}/><input placeholder="搜索组件"/></div><div className="component-group"><h4>Executors</h4>{components.slice(0,3).map(({name,desc,icon:Icon,bg,color}) => <div className="component-item" key={name}><span className="component-symbol" style={{background:bg,color}}><Icon size={15}/></span><div><div className="component-name">{name}</div><div className="component-desc">{desc}</div></div></div>)}<h4>Orchestration</h4>{components.slice(3).map(({name,desc,icon:Icon,bg,color}) => <div className="component-item" key={name}><span className="component-symbol" style={{background:bg,color}}><Icon size={15}/></span><div><div className="component-name">{name}</div><div className="component-desc">{desc}</div></div></div>)}</div></aside>
    <section className="canvas-column"><div className="task-composer"><div className="composer-box"><Sparkles size={16} color="#719c29"/><textarea aria-label="任务描述" value={task} onChange={e => setTask(e.target.value)}/><button className="primary-button lime" onClick={plan} disabled={planning}>{planning ? <LoaderCircle className="spin" size={14}/> : <GitBranch size={14}/>}自动规划</button><button className="primary-button" onClick={run}><Play size={13}/>运行</button></div></div><div className="workflow-canvas"><div className="canvas-toolbar"><button className="canvas-chip active">动态工作流</button><button className="canvas-chip">能力匹配 92%</button></div><WorkflowCanvas selectedId={selectedId} onSelect={setSelectedId} runToken={runToken}/></div></section>
    <aside className="config-panel"><div className="studio-panel-title"><h3>节点配置</h3><span className="status-pill live">已验证</span></div><div className="config-content"><div className="config-section"><span className="field-label">节点名称</span><input className="field-input" value={selected.label} readOnly/><span className="field-label">主体类型</span><input className="field-input" value={selected.kind} readOnly/></div><div className="config-section"><h4>选择依据</h4><div className="reason-box">{selected.reason}</div></div><div className="config-section"><h4>能力需求与匹配</h4>{Object.entries(selected.capabilities).map(([name,value]) => <div className="cap-row" key={name}><div className="cap-head"><span>{name}</span><strong>{Math.round(value*100)}%</strong></div><div className="cap-track"><div className="cap-fill" style={{width:`${value*100}%`}}/></div></div>)}</div><div className="config-section"><h4>运行约束</h4><div className="config-tags"><span className="config-tag">最大延迟 30s</span><span className="config-tag">成本权重 .25</span><span className="config-tag">需保留轨迹</span><span className="config-tag">失败自动降级</span></div></div></div></aside>
  </div>;
}
