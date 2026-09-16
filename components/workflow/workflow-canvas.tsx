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
  type Connection,
  type Edge,
  type EdgeChange,
  type EdgeProps,
  type Node,
  type NodeChange,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Bot,
  BrainCircuit,
  Check,
  CircleEllipsis,
  Database,
  GitBranch,
  Link2,
  UserRound,
  Wrench,
} from "lucide-react";
import type { WorkflowNodeData } from "@/lib/platform-data";

const kindStyle = {
  LLM: { icon: Bot, bg: "#edf2ff", color: "#5d7cff" },
  ML: { icon: BrainCircuit, bg: "#fff3e5", color: "#d7872d" },
  Human: { icon: UserRound, bg: "#f2ecff", color: "#8b62db" },
  Tool: { icon: Wrench, bg: "#edf3ef", color: "#5f7c6b" },
  Knowledge: { icon: Database, bg: "#e9f7f0", color: "#3a8a64" },
  Adaptive: { icon: GitBranch, bg: "#edfad7", color: "#5f8d1f" },
};

const DEFAULT_NODE_WIDTH = 178;
const DEFAULT_NODE_HEIGHT = 78;

function nodeGeometry(node: Node<WorkflowNodeData>) {
  const width = node.measured?.width || node.width || DEFAULT_NODE_WIDTH;
  const height = node.measured?.height || node.height || DEFAULT_NODE_HEIGHT;
  return {
    left: node.position.x,
    right: node.position.x + width,
    top: node.position.y,
    bottom: node.position.y + height,
    centerX: node.position.x + width / 2,
    centerY: node.position.y + height / 2,
  };
}

function automaticRouteOffset(
  edge: Edge,
  nodes: Node<WorkflowNodeData>[],
): number {
  if (String(edge.data?.edge_type || "default") === "loop") return 0;
  const source = nodes.find((node) => node.id === edge.source);
  const target = nodes.find((node) => node.id === edge.target);
  if (!source || !target) return 0;

  const sourceBox = nodeGeometry(source);
  const targetBox = nodeGeometry(target);
  const sourceX = sourceBox.right;
  const sourceY = sourceBox.centerY;
  const targetX = targetBox.left;
  const targetY = targetBox.centerY;

  // A backward edge needs a visible detour. Users can change it to a loop edge
  // when it represents iteration, but it must remain readable before that.
  if (targetX <= sourceX + 20) {
    return -Math.min(104, 58 + Math.abs(targetX - sourceX) * 0.045);
  }

  const between = nodes.filter((node) => {
    if (node.id === source.id || node.id === target.id) return false;
    const box = nodeGeometry(node);
    return box.centerX > sourceX && box.centerX < targetX;
  });
  if (!between.length) return 0;

  const blockers = between.filter((node) => {
    const box = nodeGeometry(node);
    const progress = (box.centerX - sourceX) / (targetX - sourceX);
    const lineY = sourceY + (targetY - sourceY) * progress;
    return lineY >= box.top - 24 && lineY <= box.bottom + 24;
  });

  if (!blockers.length) return 0;

  // Only nodes that really intersect the direct corridor trigger a bypass.
  // A node merely located between the endpoints but above/below the line must
  // not turn an otherwise clear forward connection into a large arch.
  const relevant = blockers;
  const averageLineDelta =
    relevant.reduce((sum, node) => {
      const box = nodeGeometry(node);
      const progress = (box.centerX - sourceX) / (targetX - sourceX);
      const lineY = sourceY + (targetY - sourceY) * progress;
      return sum + (box.centerY - lineY);
    }, 0) / relevant.length;
  const direction = averageLineDelta >= 0 ? -1 : 1;
  const magnitude = Math.min(
    122,
    50 + relevant.length * 13 + Math.abs(targetX - sourceX) * 0.028,
  );
  return direction * magnitude;
}

function roundedLanePath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
  laneY: number,
): string {
  const sourceOuterX = sourceX + 34;
  const targetOuterX = targetX - 34;
  const horizontalDirection = targetX >= sourceX ? 1 : -1;
  const sourceDirection = laneY < sourceY ? -1 : 1;
  const targetDirection = targetY < laneY ? -1 : 1;
  const radius = Math.max(
    12,
    Math.min(
      24,
      Math.abs(laneY - sourceY) / 2,
      Math.abs(targetY - laneY) / 2,
    ),
  );
  return [
    `M ${sourceX} ${sourceY}`,
    `C ${sourceX + 18} ${sourceY}, ${sourceOuterX} ${sourceY}, ${sourceOuterX} ${sourceY + sourceDirection * radius}`,
    `L ${sourceOuterX} ${laneY - sourceDirection * radius}`,
    `Q ${sourceOuterX} ${laneY}, ${sourceOuterX + horizontalDirection * radius} ${laneY}`,
    `L ${targetOuterX - horizontalDirection * radius} ${laneY}`,
    `Q ${targetOuterX} ${laneY}, ${targetOuterX} ${laneY + targetDirection * radius}`,
    `L ${targetOuterX} ${targetY - targetDirection * radius}`,
    `Q ${targetOuterX} ${targetY}, ${targetX} ${targetY}`,
  ].join(" ");
}

const WorkflowNode = memo(function WorkflowNode({ data, selected }: NodeProps) {
  const node = data as WorkflowNodeData;
  const style = kindStyle[node.kind];
  const Icon = style.icon;
  return (
    <div
      className={`flow-node ${node.kind === "Adaptive" ? "adaptive" : ""} ${selected ? "selected" : ""}`}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="workflow-port workflow-port-input"
        title="输入端口：拖入或点击完成连接"
      />
      <div className="flow-node-header">
        <span
          className="flow-node-icon"
          style={{ background: style.bg, color: style.color }}
        >
          <Icon size={14} />
        </span>
        <div>
          <div className="flow-node-title">{node.label}</div>
          <div className="flow-node-type">{node.subtitle}</div>
        </div>
      </div>
      <div className="flow-node-footer">
        <span>
          {node.state === "done" ? (
            <Check size={11} />
          ) : node.state === "running" ? (
            <CircleEllipsis size={11} />
          ) : (
            node.kind
          )}
        </span>
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
  markerEnd,
  style,
  data,
  selected,
}: EdgeProps) {
  const edgeType = String(data?.edge_type || "default");
  const color =
    edgeType === "loop"
      ? "#d7872d"
      : edgeType === "conditional"
        ? "#7b61b7"
        : "#789085";
  let path: string;
  if (edgeType === "loop") {
    const span = Math.abs(sourceX - targetX);
    const loopY =
      Math.max(sourceY, targetY) + Math.min(112, 66 + span * 0.045);
    path = roundedLanePath(sourceX, sourceY, targetX, targetY, loopY);
  } else {
    const deltaX = targetX - sourceX;
    const deltaY = targetY - sourceY;
    const routeOffset = Number(data?.route_offset || 0);
    if (Math.abs(routeOffset) < 1 && Math.abs(deltaY) <= 6) {
      path = `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`;
    } else if (Math.abs(routeOffset) < 1) {
      const direction = deltaX >= 0 ? 1 : -1;
      const controlDistance = Math.max(
        24,
        Math.min(160, Math.abs(deltaX) * 0.38),
      );
      path = `M ${sourceX} ${sourceY} C ${sourceX + direction * controlDistance} ${sourceY}, ${targetX - direction * controlDistance} ${targetY}, ${targetX} ${targetY}`;
    } else {
      const laneY =
        routeOffset < 0
          ? Math.min(sourceY, targetY) + routeOffset
          : Math.max(sourceY, targetY) + routeOffset;
      path = roundedLanePath(sourceX, sourceY, targetX, targetY, laneY);
    }
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
        strokeWidth: selected ? 2.2 : Number(style?.strokeWidth || 1.6),
        strokeLinecap: "round",
        strokeLinejoin: "round",
        filter: selected
          ? "drop-shadow(0 0 2px rgb(84 112 96 / 22%))"
          : undefined,
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
    () =>
      edges.map((edge) => ({
        ...edge,
        selected: edge.id === selectedEdgeId,
        data: {
          ...edge.data,
          route_offset: automaticRouteOffset(edge, nodes),
        },
      })),
    [edges, nodes, selectedEdgeId],
  );
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => onSelect(node.id),
    [onSelect],
  );
  const onEdgeClick = useCallback(
    (_: React.MouseEvent, edge: { id: string }) => onSelectEdge(edge.id),
    [onSelectEdge],
  );

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
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.4}
      maxZoom={1.7}
      colorMode="light"
      deleteKeyCode={["Backspace", "Delete"]}
      connectOnClick
      connectionRadius={30}
      connectionLineStyle={{ stroke: "#6c9a37", strokeWidth: 2 }}
      defaultEdgeOptions={{ type: "workflow" }}
    >
      <Background gap={20} size={1} color="#dfe7e1" />
      <MiniMap
        pannable
        zoomable
        nodeColor={(node) => (node.id === selectedId ? "#8bb346" : "#dfe7e1")}
        maskColor="rgb(244 247 243 / 72%)"
        style={{ border: "1px solid #dfe7e1", borderRadius: 10 }}
      />
      <Controls showInteractive={false} />
      <Panel position="bottom-center" className="connection-hint">
        <Link2 size={12} /> 从节点右侧拖到另一节点左侧；也可依次点击两个端口
      </Panel>
    </ReactFlow>
  );
}
