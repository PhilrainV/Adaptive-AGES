"use client";

import { BrainCircuit, Plus, RefreshCcw } from "lucide-react";
import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts";
import { capabilityAgents, radarData } from "@/lib/platform-data";

export function CapabilitiesView({ notify }: { notify:(m:string)=>void }) {
  return <>
    <div className="section-heading"><div><p className="eyebrow">Capability Space</p><h2>异构能力空间</h2><p>用统一向量比较 LLM、机器模型和人类专家，并持续基于反馈校准。</p></div><button className="primary-button" onClick={() => notify("新主体登记流程已准备")}><Plus size={14}/>登记智能主体</button></div>
    <div className="capability-grid">
      <section className="panel chart-panel"><div className="panel-header"><div><h3>能力维度对比</h3><p>标准化分值 · 最近 30 天评测</p></div><button className="ghost-button" onClick={() => notify("能力评测已进入校准队列")}><RefreshCcw size={13}/>重新校准</button></div><div className="legend-row"><span className="legend-item"><i className="legend-swatch" style={{background:"#5d7cff"}}/>LLM Agent</span><span className="legend-item"><i className="legend-swatch" style={{background:"#f3a950"}}/>ML Model</span><span className="legend-item"><i className="legend-swatch" style={{background:"#9a76e8"}}/>Human Expert</span></div><div style={{height:320,width:"100%"}}><ResponsiveContainer><RadarChart data={radarData} outerRadius="70%"><PolarGrid stroke="#dfe7e1"/><PolarAngleAxis dataKey="dimension" tick={{fontSize:11,fill:"#65736b"}}/><PolarRadiusAxis domain={[0,100]} tick={false} axisLine={false}/><Tooltip contentStyle={{border:"1px solid #dfe7e1",borderRadius:10,fontSize:11}}/><Radar name="LLM Agent" dataKey="LLM" stroke="#5d7cff" fill="#5d7cff" fillOpacity={.14} strokeWidth={2}/><Radar name="ML Model" dataKey="ML" stroke="#f3a950" fill="#f3a950" fillOpacity={.12} strokeWidth={2}/><Radar name="Human Expert" dataKey="Human" stroke="#9a76e8" fill="#9a76e8" fillOpacity={.1} strokeWidth={2}/></RadarChart></ResponsiveContainer></div></section>
      <div><section className="panel"><div className="panel-header"><div><h3>主体能力排行</h3><p>当前可调度主体</p></div></div><div className="cap-agent-list">{capabilityAgents.map(agent => <div className="cap-agent-card" key={agent.name}><div className="cap-agent-top"><span className="component-symbol" style={{background:`${agent.color}18`,color:agent.color}}><BrainCircuit size={14}/></span><span className="cap-agent-name">{agent.name}</span><span className="cap-agent-kind">{agent.kind}</span></div><div className="mini-bars">{agent.values.map((value,i) => <div className="mini-bar" key={i}><span style={{width:`${value}%`,background:agent.color}}/></div>)}</div></div>)}</div></section><div className="match-card"><div className="match-title"><BrainCircuit size={16}/>多目标能力匹配</div><p>综合任务需求、主体能力、成本、时延、可靠性和用户画像生成调度分数。</p><div className="match-formula">score = α·similarity + β·reliability − γ·cost − δ·latency</div></div></div>
    </div>
  </>;
}
