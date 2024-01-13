"""LangGraph workflow service."""

import logging
import uuid
from typing import Any
from datetime import datetime
from uuid import UUID

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.core.config import settings
from app.models.schemas import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowEdge,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    NodeType,
    EdgeType,
    StreamChunk,
    CheckpointData,
    VisualizerState,
    VisualizerNode,
    VisualizerEdge,
    DebugSnapshot,
)

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for managing and executing LangGraph workflows."""

    def __init__(self) -> None:
        self._workflows: dict[UUID, WorkflowDefinition] = {}
        self._compiled_graphs: dict[UUID, Any] = {}
        self._checkpointer: SqliteSaver | None = None
        self._active_executions: dict[str, dict] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return

        settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpointer = MemorySaver()
        self._initialized = True
        logger.info("Workflow service initialized")

    # ========== Workflow Management ==========

    def create_workflow(self, workflow: WorkflowDefinition) -> UUID:
        """Create a new workflow."""
        if not self._initialized:
            self.initialize()

        # Validate workflow
        self._validate_workflow(workflow)

        self._workflows[workflow.id] = workflow
        self._compile_workflow(workflow)

        logger.info(f"Created workflow: {workflow.name} ({workflow.id})")
        return workflow.id

    def get_workflow(self, workflow_id: UUID) -> WorkflowDefinition | None:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> list[WorkflowDefinition]:
        """List all workflows."""
        return list(self._workflows.values())

    def update_workflow(self, workflow_id: UUID, workflow: WorkflowDefinition) -> bool:
        """Update an existing workflow."""
        if workflow_id not in self._workflows:
            return False

        workflow.updated_at = datetime.utcnow()
        self._workflows[workflow_id] = workflow
        self._compile_workflow(workflow)
        return True

    def delete_workflow(self, workflow_id: UUID) -> bool:
        """Delete a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            if workflow_id in self._compiled_graphs:
                del self._compiled_graphs[workflow_id]
            return True
        return False

    def _validate_workflow(self, workflow: WorkflowDefinition) -> None:
        """Validate workflow structure."""
        if not workflow.nodes:
            raise ValueError("Workflow must have at least one node")

        if not workflow.entry_point:
            # Auto-detect start node
            start_nodes = [n for n in workflow.nodes if n.type == NodeType.START]
            if start_nodes:
                workflow.entry_point = start_nodes[0].id
            else:
                raise ValueError("Workflow must have an entry point")

        # Validate edges reference valid nodes
        node_ids = {n.id for n in workflow.nodes}
        for edge in workflow.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes")

    def _compile_workflow(self, workflow: WorkflowDefinition) -> None:
        """Compile workflow to LangGraph StateGraph."""
        graph = StateGraph(dict)

        # Add nodes
        for node in workflow.nodes:
            graph.add_node(node.id, self._create_node_function(node))

        # Add edges
        for edge in workflow.edges:
            if edge.type == EdgeType.CONDITIONAL:
                graph.add_conditional_edges(
                    edge.source,
                    self._create_condition_function(edge),
                    {edge.target: edge.target},
                )
            else:
                graph.add_edge(edge.source, edge.target)

        # Set entry point
        if workflow.entry_point:
            graph.set_entry_point(workflow.entry_point)

        # Add END edges for terminal nodes
        self._add_end_edges(graph, workflow)

        # Compile with checkpointer
        compiled = graph.compile(checkpointer=self._checkpointer)
        self._compiled_graphs[workflow.id] = compiled

    def _create_node_function(self, node: WorkflowNode):
        """Create node execution function."""
        async def node_fn(state: dict) -> dict:
            logger.debug(f"Executing node: {node.id} ({node.type})")

            if node.type == NodeType.LLM:
                return await self._execute_llm_node(node, state)
            elif node.type == NodeType.TOOL:
                return await self._execute_tool_node(node, state)
            elif node.type == NodeType.CONDITIONAL:
                return await self._execute_conditional_node(node, state)
            elif node.type == NodeType.HUMAN:
                return await self._execute_human_node(node, state)
            elif node.type == NodeType.CUSTOM:
                return await self._execute_custom_node(node, state)
            else:
                return state

        return node_fn

    async def _execute_llm_node(self, node: WorkflowNode, state: dict) -> dict:
        """Execute LLM node."""
        config = node.config
        prompt = config.get("prompt", "")
        model = config.get("model", settings.default_model)
        provider = config.get("provider", settings.default_provider)
        temperature = config.get("temperature", settings.default_temperature)

        # Build messages from state
        messages = state.get("messages", [])
        if prompt:
            messages.append(HumanMessage(content=prompt))

        # Get LLM
        if provider == "openai":
            llm = ChatOpenAI(model=model, temperature=temperature)
        elif provider == "anthropic":
            llm = ChatAnthropic(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Generate response
        response = await llm.ainvoke(messages)

        # Update state
        new_state = state.copy()
        new_state["messages"] = messages + [response]
        new_state["last_output"] = response.content

        return new_state

    async def _execute_tool_node(self, node: WorkflowNode, state: dict) -> dict:
        """Execute tool node."""
        # In production, would call registered tools
        config = node.config
        tool_name = config.get("tool", "")
        tool_args = config.get("args", {})

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        # Placeholder tool execution
        result = {"tool": tool_name, "result": f"Executed {tool_name}", "args": tool_args}

        new_state = state.copy()
        new_state.setdefault("tool_results", []).append(result)
        return new_state

    async def _execute_conditional_node(self, node: WorkflowNode, state: dict) -> dict:
        """Execute conditional node."""
        config = node.config
        condition = config.get("condition", "")

        # Evaluate condition (simplified)
        try:
            result = eval(condition, {"state": state})
            state["condition_result"] = bool(result)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            state["condition_result"] = False

        return state

    async def _execute_human_node(self, node: WorkflowNode, state: dict) -> dict:
        """Execute human-in-the-loop node."""
        # In production, would pause and wait for human input
        config = node.config
        prompt = config.get("prompt", "Human input required")

        logger.info(f"Human node waiting: {prompt}")

        # For now, auto-continue
        state["human_response"] = "auto-continued"
        return state

    async def _execute_custom_node(self, node: WorkflowNode, state: dict) -> dict:
        """Execute custom node."""
        # Custom code execution would go here
        return state

    def _create_condition_function(self, edge: WorkflowEdge):
        """Create condition function for conditional edge."""
        def condition(state: dict) -> str:
            result = state.get("condition_result", False)
            return edge.target if result else END
        return condition

    def _add_end_edges(self, graph: StateGraph, workflow: WorkflowDefinition) -> None:
        """Add END edges for terminal nodes."""
        # Find nodes with no outgoing edges
        source_nodes = {e.source for e in workflow.edges}
        all_nodes = {n.id for n in workflow.nodes}
        terminal_nodes = all_nodes - source_nodes

        for node_id in terminal_nodes:
            if node_id not in self._compiled_graphs:
                graph.add_edge(node_id, END)

    # ========== Execution ==========

    async def execute(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResponse:
        """Execute a workflow."""
        if not self._initialized:
            self.initialize()

        workflow = self._workflows.get(request.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {request.workflow_id}")

        compiled = self._compiled_graphs.get(request.workflow_id)
        if not compiled:
            raise ValueError(f"Workflow not compiled: {request.workflow_id}")

        thread_id = request.thread_id or str(uuid.uuid4())
        execution_id = uuid.uuid4()

        # Prepare input state
        initial_state = {
            "messages": [],
            **request.input,
        }

        # Track execution
        self._active_executions[thread_id] = {
            "execution_id": execution_id,
            "workflow_id": request.workflow_id,
            "status": "running",
            "started_at": datetime.utcnow(),
        }

        try:
            if request.stream:
                # Streaming execution would go here
                # For now, run to completion
                pass

            # Execute
            config = {"configurable": {"thread_id": thread_id}}
            result = await compiled.ainvoke(initial_state, config=config)

            execution = WorkflowExecutionResponse(
                execution_id=execution_id,
                workflow_id=request.workflow_id,
                thread_id=thread_id,
                status="completed",
                output=result,
                completed_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.exception(f"Workflow execution failed: {e}")
            execution = WorkflowExecutionResponse(
                execution_id=execution_id,
                workflow_id=request.workflow_id,
                thread_id=thread_id,
                status="failed",
                error=str(e),
                completed_at=datetime.utcnow(),
            )
        finally:
            if thread_id in self._active_executions:
                del self._active_executions[thread_id]

        return execution

    async def stream_execute(self, request: WorkflowExecutionRequest):
        """Execute workflow with streaming."""
        # This would use LangGraph's streaming capabilities
        # For now, yield execution events
        yield StreamChunk(type="start", data={"thread_id": request.thread_id or str(uuid.uuid4())})
        
        # Placeholder for streaming implementation
        execution = await self.execute(request)
        
        yield StreamChunk(type="end", data={"execution": execution.model_dump()})

    # ========== Checkpoints ==========

    def get_checkpoints(self, thread_id: str) -> list[CheckpointData]:
        """Get checkpoints for a thread."""
        if not self._checkpointer:
            return []

        checkpoints = []
        for checkpoint in self._checkpointer.list({"configurable": {"thread_id": thread_id}}):
            checkpoints.append(CheckpointData(
                thread_id=thread_id,
                checkpoint_id=checkpoint.config.get("configurable", {}).get("checkpoint_id", ""),
                workflow_id=UUID("00000000-0000-0000-0000-000000000000"),  # Placeholder
                state=checkpoint.checkpoint,
                metadata=checkpoint.metadata,
                created_at=datetime.fromisoformat(checkpoint.metadata.get("timestamp", datetime.utcnow().isoformat())),
            ))
        return checkpoints

    def get_checkpoint(self, thread_id: str, checkpoint_id: str) -> CheckpointData | None:
        """Get a specific checkpoint."""
        if not self._checkpointer:
            return None

        checkpoint = self._checkpointer.get({"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}})
        if not checkpoint:
            return None

        return CheckpointData(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            workflow_id=UUID("00000000-0000-0000-0000-000000000000"),
            state=checkpoint.checkpoint,
            metadata=checkpoint.metadata,
            created_at=datetime.utcnow(),
        )

    # ========== Visualization ==========

    def get_visualizer_state(self, workflow_id: UUID, execution_id: UUID | None = None) -> VisualizerState:
        """Get workflow state for visualizer."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        nodes = []
        for node in workflow.nodes:
            nodes.append(VisualizerNode(
                id=node.id,
                label=node.name,
                type=node.type.value,
                status="idle",
                position=node.position or {"x": 0, "y": 0},
            ))

        edges = []
        for edge in workflow.edges:
            edges.append(VisualizerEdge(
                source=edge.source,
                target=edge.target,
                type=edge.type.value,
                status="idle",
                label=edge.label,
            ))

        return VisualizerState(
            nodes=nodes,
            edges=edges,
            active_node=None,
            execution_id=execution_id,
        )

    # ========== Debug ==========

    def get_debug_snapshot(self, thread_id: str) -> DebugSnapshot | None:
        """Get debug snapshot for a thread."""
        if thread_id not in self._active_executions:
            return None

        exec_info = self._active_executions[thread_id]
        checkpoints = self.get_checkpoints(thread_id)

        return DebugSnapshot(
            execution_id=exec_info["execution_id"],
            thread_id=thread_id,
            current_node=None,  # Would track current node
            state={},
            history=[c.state for c in checkpoints],
        )


# Global instance
workflow_service = WorkflowService()