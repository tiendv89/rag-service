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
upserts embeddings into Qdrant.

| Variable | Required | Default | Description |
|---|---|---|---|
| `QDRANT_URL` | yes | — | Qdrant instance URL |
| `WORKSPACE_ID` | yes | — | Workspace partition key used as the Qdrant collection name |
| `REPO_PATHS` | yes | — | Comma-separated absolute paths to git repos to index |
| `INDEXER_POLL_INTERVAL_SECONDS` | no | `300` | Seconds between indexing cycles |

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
      REPO_PATHS: /repos/management-repo,/repos/workflow
      INDEXER_POLL_INTERVAL_SECONDS: "60"
    volumes:
      - /path/to/local/repos:/repos:ro
    depends_on:
      - qdrant

volumes:
  qdrant_data:
```

Key points:

- Set `SERVICE=rag_server` or `SERVICE=indexer` to choose which process runs.
- Both containers share a Qdrant instance; `QDRANT_URL` must point to it.
- The indexer needs the git repos mounted as volumes so it can read file history.
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

# Run indexer locally
QDRANT_URL=http://localhost:6333 WORKSPACE_ID=dev REPO_PATHS=/path/to/repo python -m services.indexer.main
```
