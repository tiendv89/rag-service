"""
Entry point for the RAG MCP server.

Usage:
    python -m services.rag_server.main

Environment variables (all optional — see server.py for defaults):
    QDRANT_URL              URL of the Qdrant instance
    RAG_SERVER_HOST         Host to bind (default: 0.0.0.0)
    RAG_SERVER_PORT         Port to listen on (default: 8000)
    RAG_STARTUP_RETRIES     Max Qdrant connection attempts at startup
    RAG_STARTUP_RETRY_DELAY Base delay in seconds between retries
"""

import logging
import os

import uvicorn

from services.rag_server.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

if __name__ == "__main__":
    host = os.environ.get("RAG_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_SERVER_PORT", "8000"))

    app = create_app()

    uvicorn.run(app, host=host, port=port)
