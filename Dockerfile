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

CMD if [ "$SERVICE" = "indexer" ]; then \
      exec python -m services.indexer.main; \
    else \
      exec uvicorn services.rag_server.server:create_app --factory --host 0.0.0.0 --port 8000; \
    fi
