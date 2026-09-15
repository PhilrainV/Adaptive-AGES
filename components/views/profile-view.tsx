"use client";

import { useEffect, useState } from "react";
import { BrainCircuit, GitBranch, LoaderCircle, SearchCheck, Sparkles } from "lucide-react";
import { apiFetch } from "@/lib/api";

const labels: Record<string,string> = {
  programming:"编程与实现",
  ai_literacy:"AI 素养",
  domain_knowledge:"领域知识",
  workflow_design:"流程设计",
  judgement:"专业判断",
};

interface ProfileData {
  capability: Record<string,number>;
  evidence: Record<string,unknown>;
}

const pipeline = [
  {title:"问题解析 Agent",text:"识别任务目标、子任务、风险与能力需求。",icon:SearchCheck},
  {title:"测试生成 Agent",text:"用户选择“测试后规划”时，针对当前问题动态出题。",icon:Sparkles},
  {title:"能力诊断 Agent",text:"根据本次答题证据计算人的能力与置信度。",icon:BrainCircuit},
  {title:"任务规划 Agent",text:"综合任务需求和诊断结果分配人、LLM、ML 与工具。",icon:GitBranch},
];

export function ProfileView() {
  const [profile,setProfile] = useState<ProfileData | null>(null);
  const [error,setError] = useState("");

  useEffect(() => {
    void apiFetch<ProfileData>("/human-assessments/profile")
      .then(setProfile)
      .catch(reason => setError(reason instanceof Error ? reason.message : "画像加载失败"));
  }, []);

  return <>
    <div className="section-heading"><div><p className="eyebrow">Human Capability Model</p><h2>基于任务测试证据的用户画像</h2><p>这里仅展示已经诊断出的能力；用户选择“测试后规划”时才会生成能力测试，直接规划不会读取或更新画像。</p></div></div>
    <div className="profile-grid">
      <section className="panel profile-card">
        <div className="profile-identity"><div className="profile-avatar">PW</div><div><h3 className="profile-name">当前工作区用户</h3><p className="profile-role">Human Agent · 由实际测试证据更新</p></div></div>
        {!profile && <div className="config-placeholder">{error || <><LoaderCircle className="spin" size={15}/>正在读取能力画像…</>}</div>}
        {profile && <><div className="adaptive-mode"><strong>最近一次诊断证据</strong><p>{Object.keys(profile.evidence).length ? `自动规划前测试 · 综合得分 ${Math.round(Number(profile.evidence.overall || 0)*100)}% · 置信度 ${Math.round(Number(profile.evidence.confidence || 0)*100)}%` : "尚无测试证据。首次点击“自动规划”时将针对输入的问题生成能力测试。"}</p></div><div className="evidence-bars">{Object.entries(labels).map(([key,label]) => {const value=Math.round((profile.capability[key] ?? .5)*100);return <div className="evidence-row" key={key}><div><span>{label}</span><strong>{value}%</strong></div><div className="cap-track"><div className="cap-fill" style={{width:`${value}%`}}/></div></div>})}</div></>}
      </section>
      <section className="panel">
        <div className="panel-header"><div><h3>能力如何进入自动规划</h3><p>四个 Agent 各自负责一个明确阶段</p></div><BrainCircuit size={17} color="#719c29"/></div>
        <div className="profile-pipeline">{pipeline.map(({title,text,icon:Icon},index) => <div className="profile-pipeline-item" key={title}><span><Icon size={16}/></span><div><strong>{index+1}. {title}</strong><p>{text}</p></div></div>)}</div>
      </section>
    </div>
  </>;
}
