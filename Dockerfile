FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
COPY . .
RUN uv pip install --system --no-cache -e .
CMD ["uvicorn", "services.rag_server.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
