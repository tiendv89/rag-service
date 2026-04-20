# rag-service

Python monorepo that provides two services used by the agent-workflow local
environment:

| Service | Description |
|---|---|
| `rag_server` | FastMCP/SSE server exposing the `rag_query` MCP tool (default) |
| `indexer` | Background worker that watches git repos and upserts chunks into Qdrant |

A single `Dockerfile` builds both.  The `SERVICE` environment variable selects
which process starts at container launch.

---

## Services

### rag_server

Starts `uvicorn` on port 8000 and serves the MCP SSE endpoint at `/sse`.

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `RAG_SERVER_HOST` | `0.0.0.0` | Bind host |
| `RAG_SERVER_PORT` | `8000` | Listen port |
| `RAG_STARTUP_RETRIES` | `5` | Max Qdrant connection attempts at startup |
| `RAG_STARTUP_RETRY_DELAY` | `2.0` | Base delay (seconds) between retries, doubles each attempt |

### indexer

Polls watched git repos on a configurable interval, detects changed files, and
upserts embeddings into Qdrant.  Repo paths are driven by a `workspace.yaml`
file that is **mounted into the container at startup** — not passed as an env var.

| Variable | Required | Default | Description |
|---|---|---|---|
| `QDRANT_URL` | yes | — | Qdrant instance URL |
| `WORKSPACE_ID` | yes | — | Workspace partition key used as the Qdrant collection name |
| `WORKSPACE_YAML_PATH` | yes | — | Path to `workspace.yaml` mounted inside the container (e.g. `/workspace/workspace.yaml`) |
| `INDEXER_POLL_INTERVAL_SECONDS` | no | `300` | Seconds between indexing cycles |
| `SSH_PRIVATE_KEY` | no | — | Raw PEM content of an SSH private key; written to a temp file at startup; used when cloning repos not mounted on disk (k8s / cloud) |

#### workspace.yaml format

`workspace.yaml` lists the repos the indexer should watch.  The indexer uses a
**clone-or-pull** strategy per repo:

1. If `local_path` resolves to a directory that exists on the filesystem, the
   indexer uses it directly (Docker Compose volume mount path).
2. If `local_path` is absent, unset, or does not exist, the indexer clones the
   repo from `github` (SSH URL) into `/tmp/indexer-repos/<repo_id>/` at startup
   and runs `git pull` on each cycle.

```yaml
repos:
  # Mounted volume (Docker Compose / local dev)
  - id: management-repo
    local_path: /repos/management-repo
    github: git@github.com:org/management-repo.git

  # env:VAR_NAME reference — resolved at runtime; falls back to clone if unset
  - id: workflow
    local_path: env:WORKFLOW_LOCAL_PATH
    github: git@github.com:org/workflow.git

  # No local_path — always cloned from ssh_url (k8s / cloud)
  - id: rag-service
    github: git@github.com:org/rag-service.git
```

`local_path` may be a literal path or an `env:VAR_NAME` reference.  If the env
var is unset the indexer falls back to cloning from `github`.  Repos with no
`local_path` and no `github` URL are skipped with a warning.

---

## Using from another docker-compose

Both services are consumed by the **agent-workflow** local environment.  The
`docker-compose.yml` in that repo builds this repo's image and launches the two
services as separate containers:

```yaml
# agent-workflow/docker-compose.yml (excerpt)
services:

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  rag-server:
    build:
      context: ../rag-service   # path to this repo relative to agent-workflow
      dockerfile: Dockerfile
    environment:
      SERVICE: rag_server
      QDRANT_URL: http://qdrant:6333
    ports:
      - "8000:8000"
    depends_on:
      - qdrant

  indexer:
    build:
      context: ../rag-service
      dockerfile: Dockerfile
    environment:
      SERVICE: indexer
      QDRANT_URL: http://qdrant:6333
      WORKSPACE_ID: my-workspace
      WORKSPACE_YAML_PATH: /workspace/workspace.yaml
      INDEXER_POLL_INTERVAL_SECONDS: "60"
      # Optional — only needed when repos are not pre-mounted (k8s / cloud)
      # SSH_PRIVATE_KEY: <raw PEM content of SSH private key>
    volumes:
      - /path/to/workspace.yaml:/workspace/workspace.yaml:ro
      - /path/to/local/repos:/repos:ro
    depends_on:
      - qdrant

volumes:
  qdrant_data:
```

Key points:

- Set `SERVICE=rag_server` or `SERVICE=indexer` to choose which process runs.
- Both containers share a Qdrant instance; `QDRANT_URL` must point to it.
- Mount `workspace.yaml` into the indexer container and set `WORKSPACE_YAML_PATH`
  to its container path.  The file lists which repos to index.
- The indexer also needs each repo mounted as a volume so it can read file history.
- `WORKSPACE_ID` must match the ID used by the management repo so queries hit
  the correct Qdrant collection.

---

## Local development

```bash
# Install dependencies
pip install uv
uv pip install --system -r requirements.txt
uv pip install --system -e .

# Run rag_server locally (Qdrant must be reachable)
QDRANT_URL=http://localhost:6333 uvicorn services.rag_server.server:create_app --factory --host 0.0.0.0 --port 8000

# Run indexer locally (workspace.yaml must exist at the given path)
QDRANT_URL=http://localhost:6333 WORKSPACE_ID=dev WORKSPACE_YAML_PATH=/path/to/workspace.yaml python -m services.indexer.main
```
