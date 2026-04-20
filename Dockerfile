FROM python:3.12-slim
WORKDIR /app
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

CMD if [ "$SERVICE" = "indexer" ]; then \
      exec python -m services.indexer.main; \
    else \
      exec uvicorn services.rag_server.server:create_app --factory --host 0.0.0.0 --port 8000; \
    fi
