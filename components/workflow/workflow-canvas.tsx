"use client";

import { memo, useCallback, useMemo } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  Handle,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  getSmoothStepPath,
  type Connection,
  type Edge,
  type EdgeChange,
  type EdgeProps,
  type Node,
  type NodeChange,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Bot, BrainCircuit, Check, CircleEllipsis, Database, GitBranch, Link2, UserRound, Wrench } from "lucide-react";
import type { WorkflowNodeData } from "@/lib/platform-data";

const kindStyle = {
  LLM: { icon: Bot, bg: "#edf2ff", color: "#5d7cff" },
  ML: { icon: BrainCircuit, bg: "#fff3e5", color: "#d7872d" },
  Human: { icon: UserRound, bg: "#f2ecff", color: "#8b62db" },
  Tool: { icon: Wrench, bg: "#edf3ef", color: "#5f7c6b" },
  Knowledge: { icon: Database, bg: "#e9f7f0", color: "#3a8a64" },
  Adaptive: { icon: GitBranch, bg: "#edfad7", color: "#5f8d1f" },
};

const WorkflowNode = memo(function WorkflowNode({ data, selected }: NodeProps) {
  const node = data as WorkflowNodeData;
  const style = kindStyle[node.kind];
  const Icon = style.icon;
  return (
    <div className={`flow-node ${node.kind === "Adaptive" ? "adaptive" : ""} ${selected ? "selected" : ""}`}>
      <Handle
        type="target"
        position={Position.Left}
        className="workflow-port workflow-port-input"
        title="输入端口：拖入或点击完成连接"
      />
      <div className="flow-node-header">
        <span className="flow-node-icon" style={{ background: style.bg, color: style.color }}><Icon size={14}/></span>
        <div>
          <div className="flow-node-title">{node.label}</div>
          <div className="flow-node-type">{node.subtitle}</div>
        </div>
      </div>
      <div className="flow-node-footer">
        <span>{node.state === "done" ? <Check size={11}/> : node.state === "running" ? <CircleEllipsis size={11}/> : node.kind}</span>
        <span className="node-score">匹配 {Math.round(node.score * 100)}%</span>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        className="workflow-port workflow-port-output"
        title="输出端口：拖到其他节点的输入端口"
      />
    </div>
  );
});

const WorkflowEdge = memo(function WorkflowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  data,
  selected,
}: EdgeProps) {
  const edgeType = String(data?.edge_type || "default");
  const color = edgeType === "loop" ? "#d7872d" : edgeType === "conditional" ? "#7b61b7" : "#789085";
  let path: string;
  if (edgeType === "loop") {
    const span = Math.abs(sourceX - targetX);
    const loopY = Math.max(sourceY, targetY) + Math.max(90, Math.min(180, span * .22));
    path = `M ${sourceX} ${sourceY} C ${sourceX + 70} ${loopY}, ${targetX - 70} ${loopY}, ${targetX} ${targetY}`;
  } else {
    [path] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 18,
      offset: 24,
    });
  }
  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={markerEnd}
      interactionWidth={28}
      style={{
        ...style,
        stroke: color,
        strokeWidth: selected ? 2.8 : Number(style?.strokeWidth || 1.6),
        filter: selected ? "drop-shadow(0 0 3px rgb(84 112 96 / 35%))" : undefined,
      }}
    />
  );
});

interface Props {
  nodes: Node<WorkflowNodeData>[];
  edges: Edge[];
  selectedId: string;
  selectedEdgeId: string;
  onSelect: (id: string) => void;
  onSelectEdge: (id: string) => void;
  onClearSelection: () => void;
  onNodesChange: (changes: NodeChange<Node<WorkflowNodeData>>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  onConnect: (connection: Connection) => void;
}

export function WorkflowCanvas({
  nodes,
  edges,
  selectedId,
  selectedEdgeId,
  onSelect,
  onSelectEdge,
  onClearSelection,
  onNodesChange,
  onEdgesChange,
  onConnect,
}: Props) {
  const nodeTypes = useMemo(() => ({ workflow: WorkflowNode }), []);
  const edgeTypes = useMemo(() => ({ workflow: WorkflowEdge }), []);
  const visibleEdges = useMemo(
    () => edges.map(edge => ({ ...edge, selected: edge.id === selectedEdgeId })),
    [edges, selectedEdgeId],
  );
  const onNodeClick = useCallback((_: React.MouseEvent, node: { id: string }) => onSelect(node.id), [onSelect]);
  const onEdgeClick = useCallback((_: React.MouseEvent, edge: { id: string }) => onSelectEdge(edge.id), [onSelectEdge]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={visibleEdges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onNodeClick={onNodeClick}
      onEdgeClick={onEdgeClick}
      onPaneClick={onClearSelection}
      fitView
      fitViewOptions={{ padding: .2 }}
      minZoom={.4}
      maxZoom={1.7}
      colorMode="light"
      deleteKeyCode={["Backspace", "Delete"]}
      connectOnClick
      connectionRadius={30}
      connectionLineStyle={{ stroke: "#6c9a37", strokeWidth: 2 }}
      defaultEdgeOptions={{ type: "workflow" }}
    >
      <Background gap={20} size={1} color="#dfe7e1"/>
      <MiniMap
        pannable
        zoomable
        nodeColor={(node) => node.id === selectedId ? "#8bb346" : "#dfe7e1"}
        maskColor="rgb(244 247 243 / 72%)"
        style={{ border: "1px solid #dfe7e1", borderRadius: 10 }}
      />
      <Controls showInteractive={false}/>
      <Panel position="bottom-center" className="connection-hint">
        <Link2 size={12}/> 从节点右侧拖到另一节点左侧；也可依次点击两个端口
      </Panel>
    </ReactFlow>
  );
}
