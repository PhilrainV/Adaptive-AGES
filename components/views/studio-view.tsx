"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { addEdge, useEdgesState, useNodesState, type Connection, type Edge, type Node } from "@xyflow/react";
import { Bot, BrainCircuit, Database, Download, GitBranch, KeyRound, LoaderCircle, Play, Plus, Save, Search, SlidersHorizontal, Sparkles, Trash2, UserRound, Wrench } from "lucide-react";
import { WorkflowCanvas } from "@/components/workflow/workflow-canvas";
import {
  AbilityTestDialog,
  type AbilityDiagnosis,
  type AbilityQuestion,
  type PlanningAgentTrace,
} from "@/components/planning/ability-test-dialog";
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


const modelProviders = [
  { value:"openai", label:"OpenAI", baseUrl:"", model:"gpt-4.1-mini" },
  { value:"newapi", label:"NewAPI / OneAPI", baseUrl:"", model:"gpt-4.1-mini" },
  { value:"deepseek", label:"DeepSeek", baseUrl:"https://api.deepseek.com/v1", model:"deepseek-v4-flash" },
  { value:"bailian-cn", label:"百炼（中国内地）", baseUrl:"https://dashscope.aliyuncs.com/compatible-mode/v1", model:"qwen-plus" },
  { value:"bailian-intl", label:"百炼（国际）", baseUrl:"https://dashscope-intl.aliyuncs.com/compatible-mode/v1", model:"qwen-plus" },
  { value:"openai-compatible", label:"自定义兼容接口", baseUrl:"", model:"gpt-4.1-mini" },
  { value:"local", label:"本地 Ollama / vLLM", baseUrl:"http://host.docker.internal:11434/v1", model:"qwen2.5:7b" },
] as const;

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
  const edges = plan.edges.map((edge, index) => ({
    id:`edge-${index}-${edge.source}-${edge.target}`,
    source:edge.source,
    target:edge.target,
    label:edge.condition || undefined,
    animated:true,
    data:{edge_type:edge.edge_type || "default",max_iterations:edge.max_iterations || 1},
    style:edge.edge_type === "loop" ? {stroke:"#d7872d",strokeWidth:1.8,strokeDasharray:"5 4"} : undefined,
  }));
  return { nodes, edges };
}

function defaultConfig(kind: AgentKind): Record<string, unknown> {
  if (kind === "LLM") return { use_default_model:true, modality:"text", system_prompt:"你是严谨、可解释的任务执行智能体。", prompt_template:"用户输入：{input}\n上游结果：{upstream}", temperature:.2, timeout_seconds:60, retry:1 };
  if (kind === "ML") return { runtime:"python", requirements:["numpy", "scikit-learn"], code:"def run(payload, upstream):\n    # 编写你的模型加载、特征处理和预测代码\n    return {'prediction': None, 'upstream': upstream}\n", timeout_seconds:60, retry:1 };
  if (kind === "Human") return { instruction:"请检查上游结果，给出修订意见和理由。", approval_criteria:"准确、可解释并符合领域规范。", timeout_seconds:86400 };
  return { connector:"passthrough", operation:"transform", timeout_seconds:60, retry:1 };
}

interface PlanningSession {
  session_id?: string | null;
  task_id: string;
  status?: "awaiting_answers" | "diagnosed" | "completed";
  assessment_enabled?: boolean;
  questions: AbilityQuestion[];
  diagnosis?: AbilityDiagnosis | null;
  workflow?: WorkflowPlan | null;
  agent_trace: PlanningAgentTrace[];
}

interface TaskDetail {
  task: { id:string; title:string; prompt:string; status:string };
  workflow: WorkflowPlan | null;
  planning_session: PlanningSession | null;
}

export function StudioView({
  notify,
  openTaskId,
}: {
  notify:(message:string)=>void;
  openTaskId?: string | null;
}) {
  const [task,setTask] = useState("分析学生学习数据，预测学业风险，并生成个性化教学建议，最后由教师复核");
  const [selectedId,setSelectedId] = useState("");
  const [planning,setPlanning] = useState(false);
  const [running,setRunning] = useState(false);
  const [plan,setPlan] = useState<WorkflowPlan | null>(null);
  const [nodes,setNodes,onNodesChange] = useNodesState<Node<WorkflowNodeData>>([]);
  const [edges,setEdges,onEdgesChange] = useEdgesState<Edge>([]);
  const [execution,setExecution] = useState<Record<string, unknown> | null>(null);
  const [nodeApiKeys,setNodeApiKeys] = useState<Record<string,string>>({});
  const [planningSession,setPlanningSession] = useState<PlanningSession | null>(null);
  const [diagnosis,setDiagnosis] = useState<AbilityDiagnosis | null>(null);
  const [diagnosingAssessment,setDiagnosingAssessment] = useState(false);
  const [planningWorkflow,setPlanningWorkflow] = useState(false);
  const [showAgentSettings,setShowAgentSettings] = useState(false);
  const [assessmentEnabled,setAssessmentEnabled] = useState(true);
  const [agentOverridesText,setAgentOverridesText] = useState("{}");
  const customNodeCounter = useRef(0);
  const selected = useMemo(() => nodes.find(node => node.id === selectedId), [nodes, selectedId]);
  const selectedUsesDefault = selected?.data.config?.use_default_model !== false;

  const onConnect = useCallback((connection: Connection) => setEdges(current => addEdge({...connection, animated:true}, current)), [setEdges]);

  useEffect(() => {
    if (!openTaskId) return;
    void apiFetch<TaskDetail>(`/tasks/${openTaskId}`)
      .then(detail => {
        setTask(detail.task.prompt);
        if (!detail.workflow) {
          setPlan(null); setNodes([]); setEdges([]); setSelectedId("");
          if (detail.planning_session) {
            setPlanningSession(detail.planning_session);
            setDiagnosis(detail.planning_session.diagnosis || null);
            notify(detail.planning_session.diagnosis
              ? "已恢复能力诊断结果，请确认是否开始规划"
              : "已恢复该任务尚未完成的能力测试");
          } else {
            setPlanningSession(null);
            setDiagnosis(null);
            notify("该任务尚未完成能力测试和自动规划");
          }
          return;
        }
        const visual = layoutPlan(detail.workflow);
        setPlan(detail.workflow); setNodes(visual.nodes); setEdges(visual.edges);
        setSelectedId(visual.nodes[0]?.id || ""); setPlanningSession(null); setDiagnosis(null);
        notify(`已打开任务：${detail.task.title}`);
      })
      .catch(error => notify(error instanceof Error ? error.message : "任务读取失败"))
      .finally(() => setPlanning(false));
  }, [notify, openTaskId, setEdges, setNodes]);

  const buildPlan = async () => {
    if (task.trim().length < 8) return notify("请先输入更完整的任务需求");
    let agentOverrides: Record<string,unknown> = {};
    try {
      agentOverrides = JSON.parse(agentOverridesText) as Record<string,unknown>;
      if (!agentOverrides || Array.isArray(agentOverrides) || typeof agentOverrides !== "object") throw new Error();
    } catch {
      return notify("Agent 高级配置必须是合法的 JSON 对象");
    }
    setPlanning(true);
    setExecution(null);
    setDiagnosis(null);
    try {
      const session = await apiFetch<PlanningSession>("/planning-sessions/start", {
        method:"POST",
        body:JSON.stringify({
          prompt:task,
          constraints:{},
          assessment_enabled:assessmentEnabled,
          capability_space:defaultCapabilitySpace,
          agent_overrides:agentOverrides,
        }),
      });
      if (!session.assessment_enabled && session.workflow) {
        const visual = layoutPlan(session.workflow);
        setPlan(session.workflow); setNodes(visual.nodes); setEdges(visual.edges);
        setSelectedId(visual.nodes[0]?.id || "");
        setPlanningSession(null);
        notify(`已跳过测试和用户画像，直接生成 ${visual.nodes.length} 个协同节点`);
        return;
      }
      setPlanningSession(session);
      notify("问题解析与动态出题已完成，请完成能力测试");
    } catch (error) { notify(error instanceof Error ? error.message : "自动规划失败"); }
    finally { setPlanning(false); }
  };

  const submitAssessment = async (answers: Record<string,number>) => {
    if (!planningSession?.session_id) return;
    setDiagnosingAssessment(true);
    try {
      const result = await apiFetch<{
        diagnosis:AbilityDiagnosis;
        agent_trace:PlanningAgentTrace[];
      }>(`/planning-sessions/${planningSession.session_id}/diagnose`, {
        method:"POST",
        body:JSON.stringify({answers}),
      });
      setDiagnosis(result.diagnosis);
      setPlanningSession(current => current ? {
        ...current,
        status:"diagnosed",
        diagnosis:result.diagnosis,
        agent_trace:result.agent_trace,
      } : current);
      notify(`能力诊断已完成（${Math.round(result.diagnosis.overall*100)}%），尚未生成工作流`);
    } catch (error) { notify(error instanceof Error ? error.message : "能力诊断失败"); }
    finally { setDiagnosingAssessment(false); }
  };

  const startPlanningFromDiagnosis = async () => {
    if (!planningSession?.session_id || !diagnosis) return;
    setPlanningWorkflow(true);
    try {
      const result = await apiFetch<{
        diagnosis:AbilityDiagnosis;
        workflow:WorkflowPlan;
        agent_trace:PlanningAgentTrace[];
      }>(`/planning-sessions/${planningSession.session_id}/plan`, {
        method:"POST",
        body:JSON.stringify({capability_space:defaultCapabilitySpace}),
      });
      const visual = layoutPlan(result.workflow);
      setPlan(result.workflow); setNodes(visual.nodes); setEdges(visual.edges);
      setSelectedId(visual.nodes[0]?.id || "");
      setPlanningSession(null); setDiagnosis(null);
      notify(`已根据诊断结果生成 ${visual.nodes.length} 个协同节点`);
    } catch (error) { notify(error instanceof Error ? error.message : "任务规划失败"); }
    finally { setPlanningWorkflow(false); }
  };

  const cancelAssessment = async () => {
    if (!planningSession?.session_id || diagnosingAssessment || planningWorkflow) return;
    if (diagnosis) {
      setPlanningSession(null);
      setDiagnosis(null);
      notify("诊断结果已保存，可从总览任务中继续规划");
      return;
    }
    const sessionId = planningSession.session_id;
    setPlanningSession(null);
    try {
      await apiFetch<{deleted:boolean}>(`/planning-sessions/${sessionId}`, {method:"DELETE"});
      notify("已取消本次能力测试，未创建任务");
    } catch (error) { notify(error instanceof Error ? error.message : "取消测试失败"); }
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
      const saved = await apiFetch<WorkflowPlan>(`/workflows/${plan.id}`, { method:"PUT", body:JSON.stringify({nodes:toPlanNodes(),edges:edges.map(edge => ({source:edge.source,target:edge.target,condition:typeof edge.label === "string" ? edge.label : null,edge_type:String(edge.data?.edge_type || "default"),max_iterations:Number(edge.data?.max_iterations || 1)}))}) });
      const llmNodes = nodes.filter(node => node.data.kind === "LLM");
      const nodeSettings: Array<{id:string;configured:boolean}> = [];
      for (const node of llmNodes) {
        const config = node.data.config || {};
        const result = await apiFetch<{api_key_configured:boolean}>(`/workflows/${saved.id}/nodes/${node.id}/model-settings`, {
          method:"PUT",
          body:JSON.stringify({
            use_default:config.use_default_model !== false,
            provider:String(config.provider || "openai-compatible"),
            model:String(config.model || "gpt-4.1-mini"),
            base_url:config.base_url || null,
            api_key:nodeApiKeys[node.id] || null,
            temperature:Number(config.temperature ?? .2),
            modality:String(config.modality || "text"),
          }),
        });
        nodeSettings.push({id:node.id,configured:result.api_key_configured});
      }
      setNodeApiKeys({});
      setNodes(current => current.map(node => {
        const result = nodeSettings.find(item => item.id === node.id);
        return result ? {...node,data:{...node.data,config:{...(node.data.config || {}),api_key_configured:result.configured}}} : node;
      }));
      setPlan(saved); notify("节点、连线、提示词与模型配置已保存"); return saved;
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
  const updateConfigValues = (changes: Record<string,unknown>) => selected && updateSelected({config:{...(selected.data.config || {}),...changes}});
  const updateNodeProvider = (provider:string) => {
    const preset = modelProviders.find(item => item.value === provider);
    updateConfigValues({provider,base_url:preset?.baseUrl || "",model:preset?.model || "gpt-4.1-mini"});
  };
  const deleteSelected = () => {
    if (!selected) return;
    setNodes(current => current.filter(node => node.id !== selectedId));
    setEdges(current => current.filter(edge => edge.source !== selectedId && edge.target !== selectedId));
    setSelectedId(""); notify("节点及相关连线已删除");
  };

  return <><div className="studio-shell">
    <aside className="component-panel"><div className="studio-panel-title"><h3>智能主体</h3><SlidersHorizontal size={14}/></div><div className="component-search"><Search size={14}/><input placeholder="搜索组件"/></div><div className="component-group"><h4>点击添加到画布</h4>{components.map(({name,kind,desc,icon:Icon,bg,color}) => <button className="component-item component-button" key={name} onClick={() => addNode(kind)}><span className="component-symbol" style={{background:bg,color}}><Icon size={15}/></span><div><div className="component-name">{name}</div><div className="component-desc">{desc}</div></div><Plus size={13} className="component-plus"/></button>)}</div></aside>
    <section className="canvas-column"><div className="task-composer"><div className="composer-box"><Sparkles size={16} color="#719c29"/><textarea aria-label="任务描述" value={task} onChange={event => setTask(event.target.value)}/><button className="primary-button lime" onClick={buildPlan} disabled={planning || running}>{planning ? <LoaderCircle className="spin" size={14}/> : <GitBranch size={14}/>}自动规划</button><button className="primary-button" onClick={run} disabled={running || !plan}>{running ? <LoaderCircle className="spin" size={14}/> : <Play size={13}/>}运行并导出</button></div><div className="planning-mode-selector" role="group" aria-label="能力测试模式"><button type="button" className={!assessmentEnabled ? "active" : ""} onClick={() => setAssessmentEnabled(false)} disabled={planning}><strong>直接规划</strong><span>跳过测试，不使用用户画像</span></button><button type="button" className={assessmentEnabled ? "active" : ""} onClick={() => setAssessmentEnabled(true)} disabled={planning}><strong>测试后规划</strong><span>DINA 诊断后个性化规划</span></button></div><button className="agent-settings-toggle" type="button" onClick={() => setShowAgentSettings(value => !value)}><SlidersHorizontal size={12}/>{showAgentSettings ? "收起 Agent 高级设置" : "规划 Agent 高级设置（提示词 / Skills / 算法参数）"}</button>{showAgentSettings && <div className="agent-settings-editor"><div><strong>本次规划的 Agent 覆盖配置</strong><span>JSON 会随规划会话保存，四个 Agent 可分别设置 system_prompt、skills 和 parameters。</span></div><textarea value={agentOverridesText} onChange={event => setAgentOverridesText(event.target.value)} spellCheck={false} aria-label="规划 Agent 高级 JSON 配置"/></div>}</div><div className="workflow-canvas"><div className="canvas-toolbar"><button className="canvas-chip active">{plan ? `${nodes.length} 节点动态工作流` : "输入任务后自动规划"}</button><button className="canvas-chip" onClick={() => void save()}><Save size={11}/>保存</button>{plan && <button className="canvas-chip" onClick={() => void downloadBundle(plan.id)}><Download size={11}/>导出</button>}</div><WorkflowCanvas nodes={nodes} edges={edges} selectedId={selectedId} onSelect={setSelectedId} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}/>{!nodes.length && <div className="canvas-empty"><GitBranch size={28}/><strong>等待生成动态工作流</strong><span>输入任务后，系统将分析能力需求并选择 LLM、ML、工具或人类。</span></div>}</div></section>
    <aside className="config-panel"><div className="studio-panel-title"><h3>节点配置</h3>{selected && <button className="danger-icon" onClick={deleteSelected} title="删除节点"><Trash2 size={14}/></button>}</div>{selected ? <div className="config-content"><div className="config-section"><span className="field-label">节点名称</span><input className="field-input" value={selected.data.label} onChange={event => updateSelected({label:event.target.value})}/><span className="field-label">主体类型</span><select className="field-input" value={selected.data.kind} onChange={event => {const kind=event.target.value as AgentKind;updateSelected({kind,config:defaultConfig(kind)})}}>{["LLM","ML","Human","Tool","Knowledge","Adaptive"].map(kind => <option key={kind}>{kind}</option>)}</select></div><div className="config-section"><h4>选择依据（可修订）</h4><textarea className="field-textarea" value={selected.data.reason} onChange={event => updateSelected({reason:event.target.value})}/></div>{selected.data.kind === "LLM" && <div className="config-section"><h4>LLM 模型与提示词</h4><span className="field-label">模型作用域</span><div className="model-scope-toggle"><button className={selectedUsesDefault ? "active" : ""} onClick={() => updateConfig("use_default_model",true)}>继承系统默认</button><button className={!selectedUsesDefault ? "active" : ""} onClick={() => updateConfigValues({use_default_model:false,provider:selected.data.config?.provider || "openai-compatible",model:selected.data.config?.model || "gpt-4.1-mini",base_url:selected.data.config?.base_url || "",modality:selected.data.config?.modality || "text"})}>此节点独立配置</button></div>{!selectedUsesDefault && <><span className="field-label">接口类型</span><select className="field-input" value={String(selected.data.config?.provider || "openai-compatible")} onChange={event => updateNodeProvider(event.target.value)}>{modelProviders.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select><span className="field-label">模型名称 / 微调模型 ID</span><input className="field-input" value={String(selected.data.config?.model || "")} onChange={event => updateConfig("model",event.target.value)} placeholder="模型名或你的 fine-tuned model ID"/><span className="field-label">模型能力</span><select className="field-input" value={String(selected.data.config?.modality || "text")} onChange={event => updateConfig("modality",event.target.value)}><option value="text">文本 / 推理</option><option value="vision">视觉理解（读取图像 URL）</option></select>{String(selected.data.config?.modality || "text") === "vision" && <><span className="field-label">测试图像 URL（每行一个）</span><textarea className="field-textarea" value={Array.isArray(selected.data.config?.image_urls) ? (selected.data.config?.image_urls as string[]).join("\n") : ""} onChange={event => updateConfig("image_urls",event.target.value.split("\n").map(value => value.trim()).filter(Boolean))} placeholder="https://example.com/image.jpg"/></>}<span className="field-label">Base URL</span><input className="field-input" value={String(selected.data.config?.base_url || "")} onChange={event => updateConfig("base_url",event.target.value)} placeholder="https://你的网关域名/v1"/><span className="field-label">节点 API Key {Boolean(selected.data.config?.api_key_configured) && "（留空保持原 Key）"}</span><div className="secret-input"><KeyRound size={14}/><input type="password" value={nodeApiKeys[selected.id] || ""} onChange={event => setNodeApiKeys(current => ({...current,[selected.id]:event.target.value}))} placeholder={selected.data.config?.api_key_configured ? "••••••••••••••••" : "sk-..."}/></div><p className="field-help">Key 单独加密保存；运行时此节点配置优先于系统默认模型。</p></>}<span className="field-label">System Prompt</span><textarea className="code-editor" value={String(selected.data.config?.system_prompt || "")} onChange={event => updateConfig("system_prompt",event.target.value)}/><span className="field-label">Prompt Template</span><textarea className="code-editor" value={String(selected.data.config?.prompt_template || "")} onChange={event => updateConfig("prompt_template",event.target.value)}/><span className="field-label">Temperature</span><input className="field-input" type="number" min="0" max="2" step="0.1" value={Number(selected.data.config?.temperature ?? .2)} onChange={event => updateConfig("temperature",Number(event.target.value))}/></div>}{selected.data.kind === "ML" && <div className="config-section"><h4>Python ML 代码</h4><textarea className="code-editor tall" spellCheck={false} value={String(selected.data.config?.code || "")} onChange={event => updateConfig("code",event.target.value)}/><p className="field-help">导出包中执行。代码必须定义 run(payload, upstream)。平台服务端不会直接运行任意代码。</p></div>}{selected.data.kind === "Human" && <div className="config-section"><h4>人工任务</h4><span className="field-label">操作说明</span><textarea className="field-textarea" value={String(selected.data.config?.instruction || "")} onChange={event => updateConfig("instruction",event.target.value)}/><span className="field-label">通过标准</span><textarea className="field-textarea" value={String(selected.data.config?.approval_criteria || "")} onChange={event => updateConfig("approval_criteria",event.target.value)}/></div>}<div className="config-section"><h4>能力需求与匹配</h4>{Object.entries(selected.data.capabilities).map(([name,value]) => <div className="cap-row" key={name}><div className="cap-head"><span>{name}</span><strong>{Math.round(value*100)}%</strong></div><div className="cap-track"><div className="cap-fill" style={{width:`${value*100}%`}}/></div></div>)}</div>{execution && <div className="config-section"><h4>最近运行结果</h4><pre className="execution-output">{JSON.stringify(execution,null,2)}</pre></div>}</div> : <div className="config-placeholder">选择节点后可修改名称、类型、提示词、代码和人工指令。</div>}</aside>
  </div>{planningSession && <AbilityTestDialog
    questions={planningSession.questions}
    trace={planningSession.agent_trace}
    diagnosis={diagnosis}
    diagnosing={diagnosingAssessment}
    planning={planningWorkflow}
    onCancel={() => void cancelAssessment()}
    onSubmitAnswers={answers => void submitAssessment(answers)}
    onStartPlanning={() => void startPlanningFromDiagnosis()}
  />}</>;
}
