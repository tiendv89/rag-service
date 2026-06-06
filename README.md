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
| `WORKSPACE_URL` | yes¹ | — | SSH (or HTTPS) URL of the workspace management repo. The indexer clones it and reads `workspace.yaml` from inside the clone — no host bind mount needed (mirrors the GitNexus indexer) |
| `WORKSPACE_YAML_PATH` | yes¹ | — | Path to a pre-mounted `workspace.yaml` (fallback for local dev / tests when `WORKSPACE_URL` is unset) |
| `WORKSPACE_CLONE_DIR` | no | `/tmp/indexer-workspace` | Directory the workspace repo is cloned into |
| `INDEXER_POLL_INTERVAL_SECONDS` | no | `300` | Seconds between indexing cycles |
| `SSH_PRIVATE_KEY` | no | — | Raw PEM content of an SSH private key; written to a temp file at startup; used to clone the workspace repo and any repos not mounted on disk (k8s / cloud) |

¹ Provide **either** `WORKSPACE_URL` (preferred) **or** `WORKSPACE_YAML_PATH`. `WORKSPACE_URL` takes precedence when both are set.

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
      # Clone the management repo to read workspace.yaml (no host bind mount).
      WORKSPACE_URL: git@github.com:org/project-workspace.git
      SSH_PRIVATE_KEY: <raw PEM content of SSH private key>
      INDEXER_POLL_INTERVAL_SECONDS: "60"
    depends_on:
      - qdrant

volumes:
  qdrant_data:
```

Key points:

- Set `SERVICE=rag_server` or `SERVICE=indexer` to choose which process runs.
- Both containers share a Qdrant instance; `QDRANT_URL` must point to it.
- Set `WORKSPACE_URL` to the management repo's git URL. The indexer clones it and
  reads `workspace.yaml` from inside the clone — no host bind mount required. The
  file lists which repos to index, and each is cloned on demand using the same
  `SSH_PRIVATE_KEY`. (For local dev/tests you may instead mount a `workspace.yaml`
  and set `WORKSPACE_YAML_PATH`.)
- `WORKSPACE_ID` must match the ID used by the management repo so queries hit
  the correct Qdrant collection.

---

## Operator procedures

### Qdrant collection migration (v1 → v2 embedding model upgrade)

The v2 embedding model (`BAAI/bge-base-en-v1.5`, 768-dim) is incompatible with
the v1 collection (384-dim).  The existing collection must be dropped and
recreated before deploying the new image.

1. **Stop the indexer** — ensure no writes are in flight.
2. **Drop the collection** — call the Qdrant REST API:
   ```bash
   curl -X DELETE http://<QDRANT_HOST>:6333/collections/<WORKSPACE_ID>
   ```
3. **Deploy the new indexer image** — the updated `qdrant_init.py` recreates the
   collection at 768-dim on startup.
4. **Wait one poll cycle** — the indexer detects no `_last_commit` file and
   performs a full re-index.  All content is restored within one cycle.

If the indexer starts against an existing 384-dim collection it will log a clear
error and exit rather than silently writing mismatched vectors.

---

## Local development

```bash
# Install dependencies
pip install uv
uv pip install --system -r requirements.txt
uv pip install --system -e .

# Run rag_server locally (Qdrant must be reachable)
QDRANT_URL=http://localhost:6333 uvicorn services.rag_server.server:create_app --factory --host 0.0.0.0 --port 8000

# Run indexer locally — clone the management repo (preferred)
QDRANT_URL=http://localhost:6333 WORKSPACE_ID=dev WORKSPACE_URL=git@github.com:org/project-workspace.git python -m services.indexer.main

# …or point at a local workspace.yaml (dev/test fallback)
QDRANT_URL=http://localhost:6333 WORKSPACE_ID=dev WORKSPACE_YAML_PATH=/path/to/workspace.yaml python -m services.indexer.main
```
