"""Tests for LangGraph Workflows."""

import pytest
from uuid import UUID, uuid4
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.models.schemas import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowEdge,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    NodeType,
    EdgeType,
    HealthResponse,
    VisualizerState,
    VisualizerNode,
    VisualizerEdge,
)
from app.services.workflow_service import WorkflowService


class TestConfig:
    """Test configuration loading."""

    def test_settings_defaults(self):
        """Test default settings values."""
        assert settings.app_name == "LangGraph Workflows"
        assert settings.port == 8004
        assert settings.default_model == "gpt-4o-mini"
        assert settings.stream_tokens is True


class TestSchemas:
    """Test Pydantic schemas."""

    def test_workflow_node(self):
        """Test workflow node model."""
        node = WorkflowNode(
            id="test_node",
            type=NodeType.LLM,
            name="Test LLM",
        )
        assert node.id == "test_node"
        assert node.type == NodeType.LLM

    def test_workflow_edge(self):
        """Test workflow edge model."""
        edge = WorkflowEdge(
            source="node1",
            target="node2",
            type=EdgeType.NORMAL,
        )
        assert edge.source == "node1"
        assert edge.target == "node2"

    def test_workflow_definition(self):
        """Test workflow definition model."""
        workflow = WorkflowDefinition(
            name="Test Workflow",
            nodes=[
                WorkflowNode(id="start", type=NodeType.START, name="Start"),
                WorkflowNode(id="llm", type=NodeType.LLM, name="LLM"),
            ],
            edges=[
                WorkflowEdge(source="start", target="llm"),
            ],
            entry_point="start",
        )
        assert workflow.name == "Test Workflow"
        assert len(workflow.nodes) == 2
        assert len(workflow.edges) == 1
        assert workflow.entry_point == "start"

    def test_health_response(self):
        """Test health response model."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            langgraph_version="0.2.0",
            active_executions=5,
            checkpoints_stored=100,
        )
        assert health.status == "healthy"
        assert health.active_executions == 5

    def test_visualizer_state(self):
        """Test visualizer state model."""
        state = VisualizerState(
            nodes=[
                VisualizerNode(id="n1", label="Start", type="start", status="idle", position={"x": 0, "y": 0}),
                VisualizerNode(id="n2", label="LLM", type="llm", status="idle", position={"x": 100, "y": 0}),
            ],
            edges=[
                VisualizerEdge(source="n1", target="n2", type="normal", status="idle"),
            ],
            active_node=None,
        )
        assert len(state.nodes) == 2
        assert len(state.edges) == 1


class TestWorkflowService:
    """Test workflow service."""

    @pytest.fixture
    def workflow_service(self):
        """Create workflow service instance."""
        service = WorkflowService()
        service.initialize()
        return service

    def test_create_workflow(self, workflow_service):
        """Test creating a workflow."""
        workflow = WorkflowDefinition(
            name="Test Workflow",
            nodes=[
                WorkflowNode(id="start", type=NodeType.START, name="Start"),
                WorkflowNode(id="llm", type=NodeType.LLM, name="LLM"),
            ],
            edges=[
                WorkflowEdge(source="start", target="llm"),
            ],
            entry_point="start",
        )

        workflow_id = workflow_service.create_workflow(workflow)
        assert workflow_id == workflow.id

        # Verify it's stored
        stored = workflow_service.get_workflow(workflow_id)
        assert stored is not None
        assert stored.name == "Test Workflow"

    def test_create_workflow_missing_entry_point(self, workflow_service):
        """Test workflow without entry point fails."""
        workflow = WorkflowDefinition(
            name="Test",
            nodes=[WorkflowNode(id="n1", type=NodeType.LLM, name="LLM")],
            edges=[],
        )
        workflow.entry_point = None

        with pytest.raises(ValueError):
            workflow_service.create_workflow(workflow)

    def test_list_workflows(self, workflow_service):
        """Test listing workflows."""
        w1 = WorkflowDefinition(name="W1", nodes=[WorkflowNode(id="n1", type=NodeType.LLM, name="L1")], edges=[], entry_point="n1")
        w2 = WorkflowDefinition(name="W2", nodes=[WorkflowNode(id="n2", type=NodeType.LLM, name="L2")], edges=[], entry_point="n2")

        workflow_service.create_workflow(w1)
        workflow_service.create_workflow(w2)

        workflows = workflow_service.list_workflows()
        assert len(workflows) >= 2

    def test_update_workflow(self, workflow_service):
        """Test updating a workflow."""
        workflow = WorkflowDefinition(
            name="Original",
            nodes=[WorkflowNode(id="n1", type=NodeType.LLM, name="L1")],
            edges=[],
            entry_point="n1",
        )
        workflow_id = workflow_service.create_workflow(workflow)

        workflow.name = "Updated"
        workflow.version = 2

        success = workflow_service.update_workflow(workflow_id, workflow)
        assert success

        updated = workflow_service.get_workflow(workflow_id)
        assert updated.name == "Updated"
        assert updated.version == 2

    def test_delete_workflow(self, workflow_service):
        """Test deleting a workflow."""
        workflow = WorkflowDefinition(
            name="ToDelete",
            nodes=[WorkflowNode(id="n1", type=NodeType.LLM, name="L1")],
            edges=[],
            entry_point="n1",
        )
        workflow_id = workflow_service.create_workflow(workflow)

        success = workflow_service.delete_workflow(workflow_id)
        assert success

        deleted = workflow_service.get_workflow(workflow_id)
        assert deleted is None

    def test_get_visualizer_state(self, workflow_service):
        """Test getting visualizer state."""
        workflow = WorkflowDefinition(
            name="Visual Test",
            nodes=[
                WorkflowNode(id="start", type=NodeType.START, name="Start", position={"x": 0, "y": 0}),
                WorkflowNode(id="llm", type=NodeType.LLM, name="LLM", position={"x": 100, "y": 0}),
            ],
            edges=[WorkflowEdge(source="start", target="llm")],
            entry_point="start",
        )
        workflow_id = workflow_service.create_workflow(workflow)

        state = workflow_service.get_visualizer_state(workflow_id)
        assert isinstance(state, VisualizerState)
        assert len(state.nodes) == 2
        assert len(state.edges) == 1
        assert state.nodes[0].id == "start"
        assert state.edges[0].source == "start"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])