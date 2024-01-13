"""CLI for LangGraph Workflows."""

import asyncio
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from app.core.config import settings
from app.services.workflow_service import workflow_service
from app.models.schemas import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowEdge,
    NodeType,
    EdgeType,
    WorkflowExecutionRequest,
)

app = typer.Typer(help="LangGraph Workflows CLI")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8004, "--port", "-p", help="Port to bind"),
):
    """Start the API server."""
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


@app.command()
def interactive():
    """Interactive workflow builder."""
    asyncio.run(_interactive_async())


async def _interactive_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] v{settings.app_version}"))
    console.print("[dim]Commands: create, list, execute, visualize, export, import, help, quit[/dim]")
    console.print()

    workflow_service.initialize()

    while True:
        try:
            cmd = Prompt.ask("[bold green]workflow>[/bold green]").strip().lower()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                _show_help()
            elif cmd == "create":
                await _create_workflow_interactive()
            elif cmd == "list":
                _list_workflows()
            elif cmd == "execute":
                await _execute_workflow_interactive()
            elif cmd == "visualize":
                _visualize_workflow_interactive()
            elif cmd == "export":
                _export_workflow_interactive()
            elif cmd == "import":
                _import_workflow_interactive()
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
        except (KeyboardInterrupt, EOFError):
            break


def _show_help():
    console.print("[bold]Available commands:[/bold]")
    console.print("  create     - Create new workflow interactively")
    console.print("  list       - List all workflows")
    console.print("  execute    - Execute a workflow")
    console.print("  visualize  - Show workflow visualization data")
    console.print("  export     - Export workflow to JSON")
    console.print("  import     - Import workflow from JSON")
    console.print("  help       - Show this help")
    console.print("  quit       - Exit")


def _list_workflows():
    workflows = workflow_service.list_workflows()
    if not workflows:
        console.print("[yellow]No workflows[/yellow]")
        return

    table = Table(title="Workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Nodes", style="green")
    table.add_column("Edges", style="green")

    for w in workflows:
        table.add_row(str(w.id)[:8], w.name, w.description or "", str(len(w.nodes)), str(len(w.edges)))

    console.print(table)


async def _create_workflow_interactive():
    name = Prompt.ask("Workflow name")
    description = Prompt.ask("Description", default="")

    workflow = WorkflowDefinition(name=name, description=description)

    console.print("[dim]Add nodes (empty name to finish):[/dim]")
    while True:
        node_name = Prompt.ask("Node name", default="")
        if not node_name:
            break

        node_type = Prompt.ask("Type", choices=[t.value for t in NodeType], default=NodeType.LLM.value)
        node = WorkflowNode(
            name=node_name,
            type=NodeType(node_type),
            config={},
        )
        workflow.nodes.append(node)
        console.print(f"  Added: [cyan]{node.id}[:cyan] - {node_name} ({node_type})")

    console.print("[dim]Add edges (empty source to finish):[/dim]")
    while True:
        source = Prompt.ask("Source node ID", default="")
        if not source:
            break

        target = Prompt.ask("Target node ID")
        if not target:
            break

        edge_type = Prompt.ask("Type", choices=[t.value for t in EdgeType], default=EdgeType.NORMAL.value)
        condition = None
        if edge_type == EdgeType.CONDITIONAL.value:
            condition = Prompt.ask("Condition (Python expression)")

        edge = WorkflowEdge(
            source=source,
            target=target,
            type=EdgeType(edge_type),
            condition=condition,
        )
        workflow.edges.append(edge)
        console.print(f"  Added edge: [cyan]{source} -> {target}[/cyan]")

    if workflow.nodes and not workflow.entry_point:
        workflow.entry_point = workflow.nodes[0].id

    workflow_service.create_workflow(workflow)
    console.print(f"[green]Workflow created: {workflow.id}[/green]")


async def _execute_workflow_interactive():
    _list_workflows()
    workflow_id = Prompt.ask("Workflow ID to execute")
    workflow = workflow_service.get_workflow(UUID(workflow_id))
    if not workflow:
        console.print("[red]Workflow not found[/red]")
        return

    console.print("[dim]Enter input (JSON, empty for none):[/dim]")
    input_str = Prompt.ask("Input", default="{}")
    try:
        input_data = json.loads(input_str)
    except json.JSONDecodeError:
        input_data = {}

    request = WorkflowExecutionRequest(
        workflow_id=workflow_id,
        input=input_data,
        stream=False,
    )

    with console.status("[bold yellow]Executing...[/bold yellow]", spinner="dots"):
        result = await workflow_service.execute(request)

    if result.status == "completed":
        console.print(f"[green]Completed![/green]")
        console.print(Syntax(json.dumps(result.output, indent=2, default=str), "json"))
    else:
        console.print(f"[red]Failed: {result.error}[/red]")


def _visualize_workflow_interactive():
    _list_workflows()
    workflow_id = Prompt.ask("Workflow ID")
    workflow = workflow_service.get_workflow(UUID(workflow_id))
    if not workflow:
        console.print("[red]Workflow not found[/red]")
        return

    state = workflow_service.get_visualizer_state(workflow_id)
    console.print(Syntax(json.dumps(state.model_dump(), indent=2, default=str), "json"))


def _export_workflow_interactive():
    _list_workflows()
    workflow_id = Prompt.ask("Workflow ID")
    workflow = workflow_service.get_workflow(UUID(workflow_id))
    if not workflow:
        console.print("[red]Workflow not found[/red]")
        return

    filepath = Prompt.ask("Output file", default=f"workflow_{workflow_id}.json")
    Path(filepath).write_text(json.dumps(workflow.model_dump(), indent=2, default=str))
    console.print(f"[green]Exported to {filepath}[/green]")


def _import_workflow_interactive():
    filepath = Prompt.ask("Input file")
    try:
        data = json.loads(Path(filepath).read_text())
        workflow = WorkflowDefinition(**data)
        workflow_service.create_workflow(workflow)
        console.print(f"[green]Imported: {workflow.id}[/green]")
    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")


@app.command()
def health():
    """Check service health."""
    asyncio.run(_health_async())


async def _health_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Health Check"))
    import langgraph
    health = HealthResponse(
        status="healthy",
        version=settings.app_version,
        langgraph_version=langgraph.__version__,
        active_executions=len(workflow_service._active_executions),
        checkpoints_stored=0,
    )
    console.print(f"Status: [green]{health.status}[/green]")
    console.print(f"LangGraph: [cyan]{health.langgraph_version}[/cyan]")
    console.print(f"Active Executions: [cyan]{health.active_executions}[/cyan]")


@app.command()
def list():
    """List workflows."""
    _list_workflows()


@app.command()
def export(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Export workflow to JSON."""
    workflow = workflow_service.get_workflow(UUID(workflow_id))
    if not workflow:
        console.print("[red]Workflow not found[/red]")
        return

    filepath = output or f"workflow_{workflow_id}.json"
    Path(filepath).write_text(json.dumps(workflow.model_dump(), indent=2, default=str))
    console.print(f"[green]Exported to {filepath}[/green]")


@app.command()
def import_workflow(
    filepath: str = typer.Argument(..., help="JSON file to import"),
):
    """Import workflow from JSON."""
    try:
        data = json.loads(Path(filepath).read_text())
        workflow = WorkflowDefinition(**data)
        workflow_service.create_workflow(workflow)
        console.print(f"[green]Imported: {workflow.id}[/green]")
    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")


if __name__ == "__main__":
    app()