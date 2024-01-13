"""API routes for LangGraph Workflows."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Any
from uuid import UUID
import json

from app.models.schemas import (
    WorkflowDefinition,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    StreamChunk,
    CheckpointData,
    VisualizerState,
    DebugSnapshot,
    HealthResponse,
)
from app.services.workflow_service import workflow_service

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    import langgraph
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        langgraph_version=langgraph.__version__,
        active_executions=len(workflow_service._active_executions),
        checkpoints_stored=0,  # Would query checkpointer
    )


# ========== Workflow Management ==========

@router.post("/workflows", response_model=WorkflowDefinition)
async def create_workflow(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Create a new workflow."""
    try:
        workflow_service.create_workflow(workflow)
        return workflow
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflows", response_model=list[WorkflowDefinition])
async def list_workflows() -> list[WorkflowDefinition]:
    """List all workflows."""
    return workflow_service.list_workflows()


@router.get("/workflows/{workflow_id}", response_model=WorkflowDefinition)
async def get_workflow(workflow_id: UUID) -> WorkflowDefinition:
    """Get a workflow by ID."""
    workflow = workflow_service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.put("/workflows/{workflow_id}", response_model=WorkflowDefinition)
async def update_workflow(workflow_id: UUID, workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Update a workflow."""
    if workflow.id != workflow_id:
        raise HTTPException(status_code=400, detail="Workflow ID mismatch")

    success = workflow_service.update_workflow(workflow_id, workflow)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: UUID) -> dict[str, str]:
    """Delete a workflow."""
    success = workflow_service.delete_workflow(workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"status": "deleted", "workflow_id": str(workflow_id)}


# ========== Execution ==========

@router.post("/workflows/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(workflow_id: UUID, request: WorkflowExecutionRequest) -> WorkflowExecutionResponse:
    """Execute a workflow."""
    request.workflow_id = workflow_id
    try:
        return await workflow_service.execute(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflows/{workflow_id}/execute/stream")
async def stream_execute_workflow(workflow_id: UUID, request: WorkflowExecutionRequest) -> StreamingResponse:
    """Execute a workflow with streaming."""
    request.workflow_id = workflow_id

    async def generate():
        try:
            async for chunk in workflow_service.stream_execute(request):
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ========== Checkpoints ==========

@router.get("/threads/{thread_id}/checkpoints", response_model=list[CheckpointData])
async def list_checkpoints(thread_id: str) -> list[CheckpointData]:
    """List checkpoints for a thread."""
    return workflow_service.get_checkpoints(thread_id)


@router.get("/threads/{thread_id}/checkpoints/{checkpoint_id}", response_model=CheckpointData)
async def get_checkpoint(thread_id: str, checkpoint_id: str) -> CheckpointData:
    """Get a specific checkpoint."""
    checkpoint = workflow_service.get_checkpoint(thread_id, checkpoint_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return checkpoint


# ========== Visualization ==========

@router.get("/workflows/{workflow_id}/visualize", response_model=VisualizerState)
async def get_visualizer_state(workflow_id: UUID, execution_id: UUID | None = None) -> VisualizerState:
    """Get workflow state for visualizer."""
    try:
        return workflow_service.get_visualizer_state(workflow_id, execution_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ========== Debug ==========

@router.get("/threads/{thread_id}/debug", response_model=DebugSnapshot)
async def get_debug_snapshot(thread_id: str) -> DebugSnapshot:
    """Get debug snapshot for a thread."""
    snapshot = workflow_service.get_debug_snapshot(thread_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Thread not found")
    return snapshot


@router.get("/workflows/{workflow_id}/export")
async def export_workflow(workflow_id: UUID) -> dict[str, Any]:
    """Export workflow as JSON."""
    workflow = workflow_service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.model_dump()


@router.post("/workflows/import")
async def import_workflow(workflow: WorkflowDefinition) -> WorkflowDefinition:
    """Import workflow from JSON."""
    workflow_service.create_workflow(workflow)
    return workflow