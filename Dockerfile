FROM python:3.12-slim
WORKDIR /app
# git and openssh-client are required by the indexer:
#   git — for git pull, git diff, git ls-files (GitWatcher) and git clone (workspace_resolver clone fallback)
#   openssh-client — for SSH authentication when cloning repos in k8s (SSH_PRIVATE_KEY env var)
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY . .
RUN uv pip install --system --no-cache -e .

# SERVICE selects which app to run: rag_server (default) or indexer.
# Override at runtime: docker run -e SERVICE=indexer ...
ENV SERVICE=rag_server

# When running as the indexer, workspace.yaml must be mounted into the container
# and WORKSPACE_YAML_PATH must point to it.  Example docker run flags:
#   -v /path/to/workspace.yaml:/workspace/workspace.yaml:ro
#   -e WORKSPACE_YAML_PATH=/workspace/workspace.yaml
# workspace.yaml lists the repos to index; each repo must also be mounted
# so the indexer can read file history.

# PR index cursor state file. Mount a volume for persistence across container restarts.
# Without this mount, cold start triggers a full re-index on each restart (safe, upsert is idempotent).
# Example docker run flags for persistence:
#   -v pr_index_state:/app/pr_index_state.json
# Or in docker-compose:
#   volumes:
#     - pr_index_state:/app/pr_index_state.json

CMD if [ "$SERVICE" = "indexer" ]; then \
      exec python -m services.indexer.main; \
    else \
      exec uvicorn services.rag_server.server:create_app --factory --host 0.0.0.0 --port 8000; \
    fi
