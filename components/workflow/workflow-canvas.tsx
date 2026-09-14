"use client";

import { memo, useCallback, useEffect, useMemo } from "react";
import { Background, Controls, Handle, MiniMap, Position, ReactFlow, useEdgesState, useNodesState, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Bot, BrainCircuit, Check, CircleEllipsis, Database, GitBranch, UserRound, Wrench } from "lucide-react";
import { initialEdges, initialNodes, type WorkflowNodeData } from "@/lib/platform-data";

const kindStyle = {
  LLM: { icon: Bot, bg: "#edf2ff", color: "#5d7cff" }, ML: { icon: BrainCircuit, bg: "#fff3e5", color: "#d7872d" }, Human: { icon: UserRound, bg: "#f2ecff", color: "#8b62db" }, Tool: { icon: Wrench, bg: "#edf3ef", color: "#5f7c6b" }, Knowledge: { icon: Database, bg: "#e9f7f0", color: "#3a8a64" }, Adaptive: { icon: GitBranch, bg: "#edfad7", color: "#5f8d1f" },
};

const WorkflowNode = memo(function WorkflowNode({ data, selected }: NodeProps) {
  const node = data as WorkflowNodeData;
  const style = kindStyle[node.kind];
  const Icon = style.icon;
  return <div className={`flow-node ${node.kind === "Adaptive" ? "adaptive" : ""} ${selected ? "selected" : ""}`}>
    <Handle type="target" position={Position.Left}/>
    <div className="flow-node-header"><span className="flow-node-icon" style={{background:style.bg,color:style.color}}><Icon size={14}/></span><div><div className="flow-node-title">{node.label}</div><div className="flow-node-type">{node.subtitle}</div></div></div>
    <div className="flow-node-footer"><span>{node.state === "done" ? <Check size={11}/> : node.state === "running" ? <CircleEllipsis size={11}/> : node.kind}</span><span className="node-score">匹配 {Math.round(node.score*100)}%</span></div>
    <Handle type="source" position={Position.Right}/>
  </div>;
});

export function WorkflowCanvas({ selectedId, onSelect, runToken }: { selectedId: string; onSelect: (id: string) => void; runToken: number }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);
  const nodeTypes = useMemo(() => ({ workflow: WorkflowNode }), []);
  useEffect(() => {
    if (!runToken) return;
    const timers: number[] = [];
    ["ml","llm","human","output"].forEach((id,index) => {
      timers.push(window.setTimeout(() => setNodes(current => current.map(n => n.id === id ? {...n,data:{...n.data,state:"running"}} : n)), index*550));
      timers.push(window.setTimeout(() => setNodes(current => current.map(n => n.id === id ? {...n,data:{...n.data,state:"done"}} : n)), index*550+440));
    });
    return () => timers.forEach(window.clearTimeout);
  }, [runToken, setNodes]);
  const onNodeClick = useCallback((_: React.MouseEvent, node: {id:string}) => onSelect(node.id), [onSelect]);
  return <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onNodeClick={onNodeClick} fitView fitViewOptions={{padding:.18}} minZoom={.55} maxZoom={1.5} colorMode="light" defaultEdgeOptions={{style:{stroke:"#93a79b",strokeWidth:1.4},labelStyle:{fontSize:9,fill:"#6c7a72"}}}>
    <Background gap={20} size={1} color="#dfe7e1"/><MiniMap pannable zoomable nodeColor={(n) => n.id === selectedId ? "#8bb346" : "#dfe7e1"} maskColor="rgb(244 247 243 / 72%)" style={{border:"1px solid #dfe7e1",borderRadius:10}}/><Controls showInteractive={false}/>
  </ReactFlow>;
}
