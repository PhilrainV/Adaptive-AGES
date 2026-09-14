"use client";

import { useState } from "react";
import { Save, ShieldCheck } from "lucide-react";

const initial = [
  {name:"编程能力",desc:"影响代码节点、参数面板和调试信息的开放程度",value:82},
  {name:"AI 素养",desc:"影响模型选择、提示配置与不确定性解释的深度",value:76},
  {name:"领域知识",desc:"影响系统自动化程度和人类判断节点的介入时机",value:91},
  {name:"流程设计",desc:"影响是否开放高级编排、路由和失败恢复选项",value:68},
];

export function ProfileView({notify}:{notify:(m:string)=>void}) {
  const [skills,setSkills] = useState(initial);
  const update = (i:number,v:number) => setSkills(list => list.map((item,index) => index === i ? {...item,value:v} : item));
  return <>
    <div className="section-heading"><div><p className="eyebrow">User Adaptive Model</p><h2>用户能力画像</h2><p>系统据此调整自动化程度、解释深度和参数开放范围。</p></div><button className="primary-button" onClick={() => notify("用户能力画像已保存并生效")}><Save size={14}/>保存画像</button></div>
    <div className="profile-grid"><section className="panel profile-card"><div className="profile-identity"><div className="profile-avatar">PW</div><div><h3 className="profile-name">Philrain Wei</h3><p className="profile-role">项目所有者 · 教育人工智能专家</p></div></div><div className="profile-stat"><div><strong>24</strong><span>创建工作流</span></div><div><strong>186</strong><span>人类反馈</span></div><div><strong>4.8</strong><span>协同评分</span></div></div><div className="adaptive-mode"><strong>当前模式：专家协同</strong><p>开放模型选择、能力权重和执行约束；仅在高风险决策或模型分歧时要求人工介入。</p></div></section><section className="panel"><div className="panel-header"><div><h3>能力维度</h3><p>拖动滑块调整，后续可由行为数据自动更新</p></div><ShieldCheck size={16} color="#719c29"/></div><div className="slider-list">{skills.map((skill,index) => <div className="slider-item" key={skill.name}><div className="slider-head"><span className="slider-name">{skill.name}</span><span className="slider-value">{skill.value}%</span></div><p>{skill.desc}</p><input type="range" min="0" max="100" value={skill.value} onChange={e => update(index,Number(e.target.value))}/></div>)}</div></section></div>
  </>;
}
