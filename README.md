# LangGraph Workflows

A comprehensive **LangGraph workflow platform** featuring:

- 🎨 **Visual Workflow Builder** — Define state machines with nodes and edges
- ⚡ **Streaming Execution** — Real-time token streaming with SSE
- 🔍 **Time-Travel Debugging** — Checkpoint-based debugging with history replay
- 🛠️ **Visualizer** — Live workflow state visualization
- 🔧 **LangGraph Native** — Built on LangGraph for production reliability

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Local Development

```bash
cd langgraph-workflows

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Run API server
python -m app.main

# Or use CLI
python -m app.cli interactive
```

### Docker Deployment

```bash
cd docker
docker-compose up -d

# API available at http://localhost:8004/docs
```

## Features

### Visual Workflow Definition

```python
from app.models.schemas import WorkflowDefinition, WorkflowNode, WorkflowEdge, NodeType, EdgeType

workflow = WorkflowDefinition(
    name="Code Review Assistant",
    nodes=[
        WorkflowNode(id="start", type=NodeType.START, name="Start"),
        WorkflowNode(id="analyze", type=NodeType.LLM, name="Analyze Code", 
            config={"prompt": "Review this code for bugs", "model": "gpt-4o"}),
        WorkflowNode(id="conditional", type=NodeType.CONDITIONAL, name="Has Issues?",
            config={"condition": "len(state.get('issues', [])) > 0"}),
        WorkflowNode(id="fix", type=NodeType.LLM, name="Suggest Fixes",
            config={"prompt": "Suggest fixes for the issues found"}),
        WorkflowNode(id="end", type=NodeType.END, name="Complete"),
    ],
    edges=[
        WorkflowEdge(source="start", target="analyze"),
        WorkflowEdge(source="analyze", target="conditional"),
        WorkflowEdge(source="conditional", target="fix", type=EdgeType.CONDITIONAL, condition="state['condition_result']"),
        WorkflowEdge(source="conditional", target="end", type=EdgeType.CONDITIONAL, condition="not state['condition_result']"),
        WorkflowEdge(source="fix", target="end"),
    ],
    entry_point="start",
)
```

### Execute Workflow

```python
import httpx

async with httpx.AsyncClient() as client:
    # Execute workflow
    response = await client.post("http://localhost:8004/api/v1/workflows/{workflow_id}/execute", json={
        "input": {"code": "def hello(): pass"},
        "stream": True,
    })
    
    # For streaming:
    async with client.stream("POST", "http://localhost:8004/api/v1/workflows/{workflow_id}/execute/stream", json={
        "input": {"code": "..."},
        "stream": True,
    }) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                print(chunk)
```

### Streaming Execution

```python
async with httpx.AsyncClient(timeout=60) as client:
    async with client.stream("POST", "http://localhost:8004/api/v1/workflows/{workflow_id}/execute/stream", json={
        "input": {"question": "What is LangGraph?"},
        "stream": True,
    }) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if data["type"] == "token":
                    print(data["data"]["text"], end="", flush=True)
                elif data["type"] == "node_end":
                    print(f"\n[Node {data['node_id']} completed]")
```

### Time-Travel Debugging

```python
async with httpx.AsyncClient() as client:
    # Get checkpoints for a thread
    checkpoints = await client.get("http://localhost:8004/api/v1/threads/{thread_id}/checkpoints")
    
    # Get specific checkpoint
    checkpoint = await client.get("http://localhost:8004/api/v1/threads/{thread_id}/checkpoints/{checkpoint_id}")
    
    # Resume from checkpoint (would use thread_id + checkpoint_id)
```

### Visualization

```python
async with httpx.AsyncClient() as client:
    # Get visualizer state
    state = await client.get("http://localhost:8004/api/v1/workflows/{workflow_id}/visualize")
    # state contains nodes, edges, positions for rendering
```

## API Endpoints

### Workflow Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/workflows` | Create workflow |
| GET | `/api/v1/workflows` | List workflows |
| GET | `/api/v1/workflows/{id}` | Get workflow |
| PUT | `/api/v1/workflows/{id}` | Update workflow |
| DELETE | `/api/v1/workflows/{id}` | Delete workflow |

### Execution
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/workflows/{id}/execute` | Execute workflow |
| POST | `/api/v1/workflows/{id}/execute/stream` | Stream execution |

### Checkpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/threads/{thread_id}/checkpoints` | List checkpoints |
| GET | `/api/v1/threads/{thread_id}/checkpoints/{id}` | Get checkpoint |

### Visualization
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/workflows/{id}/visualize` | Get visualizer state |

### Debug
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/threads/{thread_id}/debug` | Get debug snapshot |

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `gpt-4o-mini` | Default LLM model |
| `DEFAULT_PROVIDER` | `openai` | LLM provider |
| `STREAM_TOKENS` | `true` | Enable token streaming |
| `DEBUG_CHECKPOINTS` | `true` | Enable checkpoint debugging |
| `MAX_DEBUG_HISTORY` | `1000` | Max debug history entries |

## Node Types

| Type | Description | Config |
|------|-------------|--------|
| `start` | Entry point | - |
| `end` | Exit point | - |
| `llm` | LLM call | prompt, model, provider, temperature |
| `tool` | Tool call | tool, args |
| `conditional` | Branching | condition (Python expr) |
| `human` | Human-in-loop | prompt |
| `custom` | Custom code | code |

## Project Structure

```
langgraph-workflows/
├── app/
│   ├── api/routes.py              # FastAPI routes
│   ├── cli.py                     # Typer CLI
│   ├── core/config.py             # Pydantic settings
│   ├── main.py                    # FastAPI app entry
│   ├── models/schemas.py          # Pydantic models
│   └── services/workflow_service.py # LangGraph integration
├── docker/
│   ├── Dockerfile                 # Multi-stage build
│   └── docker-compose.yml         # Full stack
├── tests/                         # Pytest suite
├── pyproject.toml                 # Project config
└── .env.example                   # Environment template
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

## License

MIT License

## Author

Yogesh Gaikwad — Yogesh.r.gaikwad@gmail.com