"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { addEdge, useEdgesState, useNodesState, type Connection, type Edge, type Node } from "@xyflow/react";
import { Bot, BrainCircuit, Database, Download, GitBranch, LoaderCircle, Play, Plus, Save, Search, SlidersHorizontal, Sparkles, Trash2, UserRound, Wrench } from "lucide-react";
import { WorkflowCanvas } from "@/components/workflow/workflow-canvas";
import { apiDownload, apiFetch, defaultCapabilitySpace, type SubjectType, type WorkflowPlan, type WorkflowPlanNode } from "@/lib/api";
import type { AgentKind, WorkflowNodeData } from "@/lib/platform-data";

const components = [
  { name:"LLM Agent", kind:"LLM" as AgentKind, desc:"推理、生成与解释", icon:Bot, bg:"#edf2ff", color:"#5d7cff" },
  { name:"ML Model", kind:"ML" as AgentKind, desc:"预测、分类与分析", icon:BrainCircuit, bg:"#fff3e5", color:"#d7872d" },
  { name:"Human Expert", kind:"Human" as AgentKind, desc:"判断、复核与反馈", icon:UserRound, bg:"#f2ecff", color:"#8b62db" },
  { name:"Adaptive Router", kind:"Adaptive" as AgentKind, desc:"按能力动态分配", icon:GitBranch, bg:"#edfad7", color:"#5f8d1f" },
  { name:"Tool", kind:"Tool" as AgentKind, desc:"API 与数据工具", icon:Wrench, bg:"#edf3ef", color:"#5f7c6b" },
  { name:"Knowledge", kind:"Knowledge" as AgentKind, desc:"知识库与长期记忆", icon:Database, bg:"#e9f7f0", color:"#3a8a64" },
];

const kindFromSubject: Record<SubjectType, AgentKind> = { llm:"LLM", ml:"ML", human:"Human", tool:"Tool" };
const subjectFromKind = (kind: AgentKind): SubjectType => kind === "LLM" ? "llm" : kind === "ML" ? "ml" : kind === "Human" ? "human" : "tool";

function layoutPlan(plan: WorkflowPlan): { nodes: Node<WorkflowNodeData>[]; edges: Edge[] } {
  const predecessors = new Map(plan.nodes.map(node => [node.id, [] as string[]]));
  plan.edges.forEach(edge => predecessors.get(edge.target)?.push(edge.source));
  const depths = new Map<string, number>();
  const depth = (id: string, path = new Set<string>()): number => {
    if (depths.has(id)) return depths.get(id)!;
    if (path.has(id)) return 0;
    const deps = predecessors.get(id) || [];
    const value = deps.length ? Math.max(...deps.map(dep => depth(dep, new Set(path).add(id)))) + 1 : 0;
    depths.set(id, value);
    return value;
  };
  const rows = new Map<number, number>();
  const nodes = plan.nodes.map(node => {
    const column = depth(node.id);
    const row = rows.get(column) || 0;
    rows.set(column, row + 1);
    return {
      id: node.id, type: "workflow", position: { x: 50 + column * 245, y: 90 + row * 150 },
      data: {
        label: node.label, kind: kindFromSubject[node.subject_type], subtitle: node.subject_id,
        score: node.match_score, reason: node.explainability.reason || "由能力匹配器自动选择。",
        capabilities: { 综合匹配: node.match_score, 语义相似: Number(node.explainability.similarity || node.match_score) },
        state: "idle", subjectId: node.subject_id, subtaskId: node.subtask_id, config: node.config,
      },
    } as Node<WorkflowNodeData>;
  });
  const edges = plan.edges.map((edge, index) => ({ id:`edge-${index}-${edge.source}-${edge.target}`, source:edge.source, target:edge.target, label:edge.condition || undefined, animated:true }));
  return { nodes, edges };
}

function defaultConfig(kind: AgentKind): Record<string, unknown> {
  if (kind === "LLM") return { system_prompt:"你是严谨、可解释的任务执行智能体。", prompt_template:"用户输入：{input}\n上游结果：{upstream}", temperature:.2, timeout_seconds:60, retry:1 };
  if (kind === "ML") return { runtime:"python", requirements:["numpy", "scikit-learn"], code:"def run(payload, upstream):\n    # 编写你的模型加载、特征处理和预测代码\n    return {'prediction': None, 'upstream': upstream}\n", timeout_seconds:60, retry:1 };
  if (kind === "Human") return { instruction:"请检查上游结果，给出修订意见和理由。", approval_criteria:"准确、可解释并符合领域规范。", timeout_seconds:86400 };
  return { connector:"passthrough", operation:"transform", timeout_seconds:60, retry:1 };
}

export function StudioView({ notify }: { notify:(message:string)=>void }) {
  const [task,setTask] = useState("分析学生学习数据，预测学业风险，并生成个性化教学建议，最后由教师复核");
  const [selectedId,setSelectedId] = useState("");
  const [planning,setPlanning] = useState(false);
  const [running,setRunning] = useState(false);
  const [plan,setPlan] = useState<WorkflowPlan | null>(null);
  const [nodes,setNodes,onNodesChange] = useNodesState<Node<WorkflowNodeData>>([]);
  const [edges,setEdges,onEdgesChange] = useEdgesState<Edge>([]);
  const [execution,setExecution] = useState<Record<string, unknown> | null>(null);
  const customNodeCounter = useRef(0);
  const selected = useMemo(() => nodes.find(node => node.id === selectedId), [nodes, selectedId]);

  const onConnect = useCallback((connection: Connection) => setEdges(current => addEdge({...connection, animated:true}, current)), [setEdges]);

  const buildPlan = async () => {
    if (task.trim().length < 8) return notify("请先输入更完整的任务需求");
    setPlanning(true);
    setExecution(null);
    try {
      const graph = await apiFetch<Record<string, unknown> & { task_id:string; subtasks:unknown[]; planning_mode:string }>("/tasks/understand", { method:"POST", body:JSON.stringify({prompt:task,constraints:{}}) });
      const profile = await apiFetch<{capability:Record<string,number>}>("/human-assessments/profile").catch(() => ({capability:{}}));
      const capabilitySpace = defaultCapabilitySpace.map(subject => subject.subject_type === "human" ? {...subject, capability:{...subject.capability,...profile.capability}} : subject);
      const generated = await apiFetch<WorkflowPlan>("/workflows/plan", { method:"POST", body:JSON.stringify({task_graph:graph,capability_space:capabilitySpace,user_profile:profile.capability}) });
      const visual = layoutPlan(generated);
      setPlan(generated); setNodes(visual.nodes); setEdges(visual.edges); setSelectedId(visual.nodes[0]?.id || "");
      notify(`已通过${graph.planning_mode === "llm" ? "模型" : "规则降级"}规划生成 ${visual.nodes.length} 个节点`);
    } catch (error) { notify(error instanceof Error ? error.message : "自动规划失败"); }
    finally { setPlanning(false); }
  };

  const toPlanNodes = (): WorkflowPlanNode[] => nodes.map(node => {
    const existing = plan?.nodes.find(item => item.id === node.id);
    return {
      id:node.id, subtask_id:node.data.subtaskId || existing?.subtask_id || node.id,
      subject_id:node.data.subjectId || existing?.subject_id || `custom-${subjectFromKind(node.data.kind)}`,
      subject_type:subjectFromKind(node.data.kind), label:node.data.label, match_score:node.data.score,
      config:node.data.config || {}, explainability:{...(existing?.explainability || {}),reason:node.data.reason},
    };
  });

  const save = async (): Promise<WorkflowPlan | null> => {
    if (!plan) { notify("请先自动规划，再保存工作流"); return null; }
    try {
      const saved = await apiFetch<WorkflowPlan>(`/workflows/${plan.id}`, { method:"PUT", body:JSON.stringify({nodes:toPlanNodes(),edges:edges.map(edge => ({source:edge.source,target:edge.target,condition:typeof edge.label === "string" ? edge.label : null}))}) });
      setPlan(saved); notify("节点、连线和配置已保存"); return saved;
    } catch (error) { notify(error instanceof Error ? error.message : "保存失败"); return null; }
  };

  const downloadBundle = async (workflowId: string) => {
    const blob = await apiDownload(`/workflows/${workflowId}/export`);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a"); anchor.href=url; anchor.download=`adaptive-workflow-${workflowId.slice(0,8)}.zip`; anchor.click();
    URL.revokeObjectURL(url);
  };

  const run = async () => {
    setRunning(true); setExecution(null);
    let saved: WorkflowPlan | null = null;
    try {
      saved = await save();
      if (!saved) return;
      setNodes(current => current.map(node => ({...node,data:{...node.data,state:"running"}})));
      const result = await apiFetch<Record<string, unknown>>(`/workflows/${saved.id}/execute`, { method:"POST", body:JSON.stringify({input:{request:task}}) });
      setExecution(result);
      setNodes(current => current.map(node => ({...node,data:{...node.data,state:"done"}})));
      await downloadBundle(saved.id);
      saved = null;
      notify(result.status === "waiting_for_human" ? "执行已暂停等待人工处理；运行包已导出" : "执行完成；可运行 ZIP 已自动下载");
    } catch (error) {
      setNodes(current => current.map(node => ({...node,data:{...node.data,state:"idle"}})));
      if (saved) {
        try { await downloadBundle(saved.id); notify(`执行未完成，但可运行包已导出：${error instanceof Error ? error.message : "执行失败"}`); }
        catch { notify(error instanceof Error ? error.message : "执行与导出均失败"); }
      } else notify(error instanceof Error ? error.message : "执行失败");
    } finally { setRunning(false); }
  };

  const addNode = (kind: AgentKind) => {
    if (!plan) return notify("请先自动规划，再添加自定义节点");
    customNodeCounter.current += 1;
    const id = `custom-node-${customNodeCounter.current}`;
    const label = kind === "LLM" ? "新 LLM Agent" : kind === "ML" ? "新 ML 节点" : kind === "Human" ? "新人工节点" : kind === "Adaptive" ? "自适应路由" : kind === "Knowledge" ? "知识节点" : "新工具节点";
    setNodes(current => [...current, {id,type:"workflow",position:{x:180+current.length*35,y:130+current.length*28},data:{label,kind,subtitle:"Custom",score:.7,reason:"由用户手动添加，可继续编辑配置与连线。",capabilities:{用户配置:.7},state:"idle",subjectId:`custom-${subjectFromKind(kind)}`,subtaskId:id,config:defaultConfig(kind)}}]);
    setSelectedId(id); notify(`已添加${label}，请在右侧配置`);
  };

  const updateSelected = (changes: Partial<WorkflowNodeData>) => setNodes(current => current.map(node => node.id === selectedId ? {...node,data:{...node.data,...changes}} : node));
  const updateConfig = (key: string, value: unknown) => selected && updateSelected({config:{...(selected.data.config || {}),[key]:value}});
  const deleteSelected = () => {
    if (!selected) return;
    setNodes(current => current.filter(node => node.id !== selectedId));
    setEdges(current => current.filter(edge => edge.source !== selectedId && edge.target !== selectedId));
    setSelectedId(""); notify("节点及相关连线已删除");
  };

  return <div className="studio-shell">
    <aside className="component-panel"><div className="studio-panel-title"><h3>智能主体</h3><SlidersHorizontal size={14}/></div><div className="component-search"><Search size={14}/><input placeholder="搜索组件"/></div><div className="component-group"><h4>点击添加到画布</h4>{components.map(({name,kind,desc,icon:Icon,bg,color}) => <button className="component-item component-button" key={name} onClick={() => addNode(kind)}><span className="component-symbol" style={{background:bg,color}}><Icon size={15}/></span><div><div className="component-name">{name}</div><div className="component-desc">{desc}</div></div><Plus size={13} className="component-plus"/></button>)}</div></aside>
    <section className="canvas-column"><div className="task-composer"><div className="composer-box"><Sparkles size={16} color="#719c29"/><textarea aria-label="任务描述" value={task} onChange={event => setTask(event.target.value)}/><button className="primary-button lime" onClick={buildPlan} disabled={planning || running}>{planning ? <LoaderCircle className="spin" size={14}/> : <GitBranch size={14}/>}自动规划</button><button className="primary-button" onClick={run} disabled={running || !plan}>{running ? <LoaderCircle className="spin" size={14}/> : <Play size={13}/>}运行并导出</button></div></div><div className="workflow-canvas"><div className="canvas-toolbar"><button className="canvas-chip active">{plan ? `${nodes.length} 节点动态工作流` : "输入任务后自动规划"}</button><button className="canvas-chip" onClick={() => void save()}><Save size={11}/>保存</button>{plan && <button className="canvas-chip" onClick={() => void downloadBundle(plan.id)}><Download size={11}/>导出</button>}</div><WorkflowCanvas nodes={nodes} edges={edges} selectedId={selectedId} onSelect={setSelectedId} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}/>{!nodes.length && <div className="canvas-empty"><GitBranch size={28}/><strong>等待生成动态工作流</strong><span>输入任务后，系统将分析能力需求并选择 LLM、ML、工具或人类。</span></div>}</div></section>
    <aside className="config-panel"><div className="studio-panel-title"><h3>节点配置</h3>{selected && <button className="danger-icon" onClick={deleteSelected} title="删除节点"><Trash2 size={14}/></button>}</div>{selected ? <div className="config-content"><div className="config-section"><span className="field-label">节点名称</span><input className="field-input" value={selected.data.label} onChange={event => updateSelected({label:event.target.value})}/><span className="field-label">主体类型</span><select className="field-input" value={selected.data.kind} onChange={event => {const kind=event.target.value as AgentKind;updateSelected({kind,config:defaultConfig(kind)})}}>{["LLM","ML","Human","Tool","Knowledge","Adaptive"].map(kind => <option key={kind}>{kind}</option>)}</select></div><div className="config-section"><h4>选择依据（可修订）</h4><textarea className="field-textarea" value={selected.data.reason} onChange={event => updateSelected({reason:event.target.value})}/></div>{selected.data.kind === "LLM" && <div className="config-section"><h4>LLM 提示词</h4><span className="field-label">System Prompt</span><textarea className="code-editor" value={String(selected.data.config?.system_prompt || "")} onChange={event => updateConfig("system_prompt",event.target.value)}/><span className="field-label">Prompt Template</span><textarea className="code-editor" value={String(selected.data.config?.prompt_template || "")} onChange={event => updateConfig("prompt_template",event.target.value)}/><span className="field-label">Temperature</span><input className="field-input" type="number" min="0" max="2" step="0.1" value={Number(selected.data.config?.temperature ?? .2)} onChange={event => updateConfig("temperature",Number(event.target.value))}/></div>}{selected.data.kind === "ML" && <div className="config-section"><h4>Python ML 代码</h4><textarea className="code-editor tall" spellCheck={false} value={String(selected.data.config?.code || "")} onChange={event => updateConfig("code",event.target.value)}/><p className="field-help">导出包中执行。代码必须定义 run(payload, upstream)。平台服务端不会直接运行任意代码。</p></div>}{selected.data.kind === "Human" && <div className="config-section"><h4>人工任务</h4><span className="field-label">操作说明</span><textarea className="field-textarea" value={String(selected.data.config?.instruction || "")} onChange={event => updateConfig("instruction",event.target.value)}/><span className="field-label">通过标准</span><textarea className="field-textarea" value={String(selected.data.config?.approval_criteria || "")} onChange={event => updateConfig("approval_criteria",event.target.value)}/></div>}<div className="config-section"><h4>能力需求与匹配</h4>{Object.entries(selected.data.capabilities).map(([name,value]) => <div className="cap-row" key={name}><div className="cap-head"><span>{name}</span><strong>{Math.round(value*100)}%</strong></div><div className="cap-track"><div className="cap-fill" style={{width:`${value*100}%`}}/></div></div>)}</div>{execution && <div className="config-section"><h4>最近运行结果</h4><pre className="execution-output">{JSON.stringify(execution,null,2)}</pre></div>}</div> : <div className="config-placeholder">选择节点后可修改名称、类型、提示词、代码和人工指令。</div>}</aside>
  </div>;
}
