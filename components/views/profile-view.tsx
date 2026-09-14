"use client";

import { useEffect, useState } from "react";
import { BrainCircuit, CheckCircle2, ClipboardCheck, LoaderCircle, Sparkles } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface Question { id:string; dimension:string; prompt:string; options:string[] }
const labels: Record<string,string> = {programming:"编程能力",ai_literacy:"AI 素养",domain_knowledge:"领域知识",workflow_design:"流程设计",judgement:"专业判断"};

export function ProfileView({notify}:{notify:(message:string)=>void}) {
  const [requirement,setRequirement] = useState("设计一个能够分析学生作答、预测学习风险、生成个性化反馈并由教师复核的智能教学系统");
  const [assessmentId,setAssessmentId] = useState("");
  const [questions,setQuestions] = useState<Question[]>([]);
  const [answers,setAnswers] = useState<Record<string,number>>({});
  const [capability,setCapability] = useState<Record<string,number>>({});
  const [evidence,setEvidence] = useState<Record<string,unknown>>({});
  const [loading,setLoading] = useState(false);

  const loadProfile = () => apiFetch<{capability:Record<string,number>;evidence:Record<string,unknown>}>("/human-assessments/profile").then(data => {setCapability(data.capability);setEvidence(data.evidence)}).catch(() => undefined);
  useEffect(() => { void loadProfile(); }, []);
  const generate = async () => {
    setLoading(true);
    try {
      const data = await apiFetch<{assessment_id:string;questions:Question[]}>("/human-assessments/generate",{method:"POST",body:JSON.stringify({design_requirement:requirement})});
      setAssessmentId(data.assessment_id); setQuestions(data.questions); setAnswers({}); notify(`已根据设计需求生成 ${data.questions.length} 道能力测试题`);
    } catch (error) { notify(error instanceof Error ? error.message : "生成失败"); }
    finally { setLoading(false); }
  };
  const submit = async () => {
    if (Object.keys(answers).length !== questions.length) return notify("请完成全部测试题");
    setLoading(true);
    try {
      const result = await apiFetch<{capability:Record<string,number>;overall:number}>(`/human-assessments/${assessmentId}/submit`,{method:"POST",body:JSON.stringify({answers})});
      setCapability(result.capability); setQuestions([]); setAssessmentId(""); await loadProfile(); notify(`评估完成，综合能力 ${Math.round(result.overall*100)}%`);
    } catch (error) { notify(error instanceof Error ? error.message : "提交失败"); }
    finally { setLoading(false); }
  };

  return <>
    <div className="section-heading"><div><p className="eyebrow">Human Capability Model</p><h2>基于任务证据的人类能力画像</h2><p>人的能力不再由手动滑块声明，而是由设计需求生成的测试及后续决策记录共同估计。</p></div></div>
    <div className="profile-grid"><section className="panel profile-card"><div className="profile-identity"><div className="profile-avatar">PW</div><div><h3 className="profile-name">当前工作区用户</h3><p className="profile-role">Human Agent · 能力随证据更新</p></div></div><div className="adaptive-mode"><strong>能力证据来源</strong><p>{Object.keys(evidence).length ? `任务自适应测试 · 综合得分 ${Math.round(Number(evidence.overall || 0)*100)}%` : "尚未完成测试；规划器暂时使用低置信度先验能力。"}</p></div><div className="evidence-bars">{Object.entries(labels).map(([key,label]) => {const value=Math.round((capability[key] ?? .5)*100);return <div className="evidence-row" key={key}><div><span>{label}</span><strong>{value}%</strong></div><div className="cap-track"><div className="cap-fill" style={{width:`${value}%`}}/></div></div>})}</div></section><section className="panel assessment-builder"><div className="panel-header"><div><h3>任务自适应能力测试</h3><p>先描述要设计的系统，平台再生成与任务相关的能力题</p></div><BrainCircuit size={17} color="#719c29"/></div><div className="assessment-compose"><label>设计需求</label><textarea value={requirement} onChange={event => setRequirement(event.target.value)} placeholder="描述你准备设计和参与的任务……"/><button className="primary-button lime" onClick={generate} disabled={loading}>{loading ? <LoaderCircle className="spin" size={14}/> : <Sparkles size={14}/>}生成能力测试</button></div></section></div>
    {!!questions.length && <section className="panel assessment-panel"><div className="panel-header"><div><h3>能力测试</h3><p>{questions.length} 道题 · 回答将更新 Human Agent 能力向量</p></div><span className="status-pill review">已答 {Object.keys(answers).length}/{questions.length}</span></div><div className="question-list">{questions.map((question,index) => <div className="question-card" key={question.id}><div className="question-title"><span>{index+1}</span><div><strong>{question.prompt}</strong><small>{labels[question.dimension] || question.dimension}</small></div></div><div className="option-list">{question.options.map((option,optionIndex) => <label className={answers[question.id] === optionIndex ? "selected" : ""} key={option}><input type="radio" name={question.id} checked={answers[question.id] === optionIndex} onChange={() => setAnswers(current => ({...current,[question.id]:optionIndex}))}/><span>{option}</span></label>)}</div></div>)}</div><div className="assessment-submit"><button className="primary-button" onClick={submit} disabled={loading}>{loading ? <LoaderCircle className="spin" size={14}/> : Object.keys(answers).length === questions.length ? <CheckCircle2 size={14}/> : <ClipboardCheck size={14}/>}提交并更新能力画像</button></div></section>}
  </>;
}
