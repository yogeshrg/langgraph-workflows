"""Pydantic models for LangGraph Workflows."""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class NodeType(str, Enum):
    """Types of workflow nodes."""

    START = "start"
    END = "end"
    LLM = "llm"
    TOOL = "tool"
    CONDITIONAL = "conditional"
    HUMAN = "human"
    CUSTOM = "custom"


class EdgeType(str, Enum):
    """Types of workflow edges."""

    NORMAL = "normal"
    CONDITIONAL = "conditional"


class WorkflowNode(BaseModel):
    """A node in the workflow graph."""

    id: str
    type: NodeType
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    position: dict[str, float] = Field(default_factory=dict)  # x, y for visualizer


class WorkflowEdge(BaseModel):
    """An edge in the workflow graph."""

    source: str
    target: str
    type: EdgeType = EdgeType.NORMAL
    condition: str | None = None  # For conditional edges
    label: str | None = None


class WorkflowDefinition(BaseModel):
    """Complete workflow definition."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    entry_point: str | None = None
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowExecutionRequest(BaseModel):
    """Request to execute a workflow."""

    workflow_id: UUID
    input: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    stream: bool = True


class WorkflowExecutionResponse(BaseModel):
    """Response from workflow execution."""

    execution_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    thread_id: str
    status: str  # running, completed, failed, paused
    output: dict[str, Any] | None = None
    error: str | None = None
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class StreamChunk(BaseModel):
    """A chunk of streaming output."""

    type: str  # node_start, node_end, tool_call, token, checkpoint
    node_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CheckpointData(BaseModel):
    """Checkpoint data for time-travel debugging."""

    thread_id: str
    checkpoint_id: str
    workflow_id: UUID
    state: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VisualizerNode(BaseModel):
    """Node data for visualizer."""

    id: str
    label: str
    type: str
    status: str  # idle, running, completed, error
    position: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualizerEdge(BaseModel):
    """Edge data for visualizer."""

    source: str
    target: str
    type: str
    status: str  # idle, active, completed
    label: str | None = None


class VisualizerState(BaseModel):
    """Complete state for visualizer."""

    nodes: list[VisualizerNode]
    edges: list[VisualizerEdge]
    active_node: str | None = None
    execution_id: UUID | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    langgraph_version: str
    active_executions: int
    checkpoints_stored: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DebugSnapshot(BaseModel):
    """Debug snapshot of execution state."""

    execution_id: UUID
    thread_id: str
    current_node: str | None
    state: dict[str, Any]
    history: list[dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)