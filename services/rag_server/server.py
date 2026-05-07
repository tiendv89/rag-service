"""
RAG MCP server — FastAPI service implementing the MCP protocol over HTTP/SSE.

Exposes one MCP tool: ``rag_query``.
Also exposes a plain HTTP endpoint: ``POST /query``.

Environment variables:
  QDRANT_URL              URL of the Qdrant instance (default: http://localhost:6333)
  RAG_SERVER_HOST         Host to bind (default: 0.0.0.0)
  RAG_SERVER_PORT         Port to listen on (default: 8000)
  RAG_STARTUP_RETRIES     Max Qdrant connection attempts at startup (default: 5)
  RAG_STARTUP_RETRY_DELAY Base delay in seconds between retries, doubles each attempt (default: 2.0)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from pydantic import BaseModel
from qdrant_client import QdrantClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from services.rag_server.embedder import Embedder
from services.shared.qdrant_init import query_points

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (populated at server startup)
# ---------------------------------------------------------------------------

_state: dict = {
    "client": None,   # QdrantClient | None
    "embedder": None, # Embedder | None
}

# ---------------------------------------------------------------------------
# Pydantic models for the REST /query endpoint
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    workspace_id: str
    top_k: int = 5
    source_types: Optional[list[str]] = None


class QueryResultItem(BaseModel):
    content: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    results: list[QueryResultItem]


# ---------------------------------------------------------------------------
# Internal retrieval function (shared by MCP tool and REST endpoint)
# ---------------------------------------------------------------------------


async def _rag_query(
    query: str,
    workspace_id: str,
    top_k: int = 5,
    source_types: Optional[list[str]] = None,
) -> list[dict]:
    """Core retrieval logic: embed query, search Qdrant, return ranked chunks."""
    if not workspace_id:
        raise ValueError(
            "workspace_id is required — queries without workspace_id are rejected"
        )

    client: Optional[QdrantClient] = _state["client"]
    embedder: Optional[Embedder] = _state["embedder"]

    if client is None:
        raise RuntimeError(
            "Qdrant is not available — check QDRANT_URL and ensure Qdrant is running"
        )
    if embedder is None:
        raise RuntimeError("Embedding model is not initialised")

    query_vector = embedder.encode(query)[0]

    results = query_points(
        client=client,
        workspace_id=workspace_id,
        query_vector=query_vector,
        top_k=top_k,
        source_types=source_types,
    )

    return [
        {
            "content": hit["payload"].get("content", ""),
            "source_path": hit["payload"].get("source_path", ""),
            "source_type": hit["payload"].get("source_type", ""),
            "feature_id": hit["payload"].get("feature_id"),
            "score": hit["score"],
        }
        for hit in results
    ]


# ---------------------------------------------------------------------------
# FastMCP server definition
# ---------------------------------------------------------------------------

mcp_server = FastMCP("rag-server", host="0.0.0.0")


@mcp_server.tool()
async def rag_query(
    query: str,
    workspace_id: str,
    top_k: int = 5,
    source_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Query the RAG index for the most relevant document chunks matching the query.

    Args:
        query: Natural language question or search text.
        workspace_id: Required tenant partition key.  Queries without a
            workspace_id are rejected with a clear error.
        top_k: Number of ranked chunks to return (default: 5).
        source_types: Optional list of source_type values to restrict results
            (skill, task_log, product_spec, technical_design, readme, claude_md).

    Returns:
        A list of ranked chunks.  Each item contains:
            content     — the text content of the chunk (empty string if not indexed)
            source_path — relative path from repo root
            source_type — one of the valid source_type values
            feature_id  — feature scope (null for workspace-wide documents)
            score       — cosine similarity score in [0, 1]
    """
    return await _rag_query(query, workspace_id, top_k, source_types)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    qdrant_url: Optional[str] = None,
    startup_retries: Optional[int] = None,
    startup_retry_delay: Optional[float] = None,
) -> Starlette:
    """
    Create and return the Starlette application that serves the MCP server over
    HTTP/SSE and the plain HTTP POST /query endpoint.

    This factory is the canonical way to build the app — uvicorn should receive
    the return value of ``create_app()``, not a bare Starlette instance.

    Startup behaviour:
    - The sentence-transformers embedding model is loaded unconditionally.
    - Qdrant connection is attempted ``startup_retries`` times with exponential
      back-off.  If Qdrant is still unreachable after all retries, the server
      starts anyway (graceful degradation) and ``rag_query`` calls will
      fail until Qdrant becomes available.

    Args:
        qdrant_url: Override QDRANT_URL env var (mainly for testing).
        startup_retries: Override RAG_STARTUP_RETRIES env var.
        startup_retry_delay: Override RAG_STARTUP_RETRY_DELAY env var.
    """
    resolved_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    max_retries = startup_retries if startup_retries is not None else int(
        os.environ.get("RAG_STARTUP_RETRIES", "5")
    )
    base_delay = startup_retry_delay if startup_retry_delay is not None else float(
        os.environ.get("RAG_STARTUP_RETRY_DELAY", "2.0")
    )

    @asynccontextmanager
    async def lifespan(app: Starlette):  # noqa: ANN001
        # --- startup ---
        # Load the embedding model first (independent of Qdrant)
        logger.info("Loading embedding model…")
        _state["embedder"] = Embedder()
        logger.info("Embedding model ready.")

        # Connect to Qdrant with exponential back-off
        for attempt in range(max_retries):
            try:
                client = QdrantClient(url=resolved_url)
                client.get_collections()  # probe — raises if unreachable
                _state["client"] = client
                logger.info("Connected to Qdrant at %s", resolved_url)
                break
            except Exception as exc:
                remaining = max_retries - attempt - 1
                if remaining > 0:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Qdrant not reachable (attempt %d/%d): %s. "
                        "Retrying in %.1f s…",
                        attempt + 1,
                        max_retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "Qdrant not reachable after %d attempt(s): %s. "
                        "Server starting without Qdrant — rag_query calls will "
                        "fail until Qdrant becomes available.",
                        max_retries,
                        exc,
                    )

        yield

        # --- shutdown ---
        _state["client"] = None
        _state["embedder"] = None
        logger.info("RAG server shut down.")

    async def health(request: Request) -> JSONResponse:
        qdrant_ok = _state["client"] is not None
        return JSONResponse(
            {
                "status": "ok",
                "qdrant": "connected" if qdrant_ok else "unavailable",
                "embedder": "ready" if _state["embedder"] is not None else "not_ready",
            },
            status_code=200,
        )

    async def query_endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=422)

        try:
            req = QueryRequest(**body)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

        try:
            hits = await _rag_query(
                query=req.query,
                workspace_id=req.workspace_id,
                top_k=req.top_k,
                source_types=req.source_types,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except Exception as exc:
            logger.error("Query endpoint error: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

        results = [
            {
                "content": hit["content"],
                "score": hit["score"],
                "metadata": {
                    "source_type": hit["source_type"],
                    "repo": None,
                    "file_path": hit["source_path"],
                    "feature_id": hit["feature_id"],
                },
            }
            for hit in hits
        ]

        return JSONResponse({"results": results})

    # Build SSE transport manually so we can pass stateless=True to run().
    # FastMCP.sse_app() hardcodes stateless=False, causing a harmless but noisy
    # warning when Claude Code sends a tool call before the MCP 'initialized'
    # notification is processed (a race that occurs on every executor startup).
    _sse = SseServerTransport("/messages/")

    async def handle_sse(scope: Scope, receive: Receive, send: Send) -> None:
        async with _sse.connect_sse(scope, receive, send) as streams:
            await mcp_server._mcp_server.run(
                streams[0],
                streams[1],
                mcp_server._mcp_server.create_initialization_options(),
                stateless=True,
            )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/query", query_endpoint, methods=["POST"]),
            Mount("/sse", app=handle_sse),
            Mount("/messages/", app=_sse.handle_post_message),
        ],
        lifespan=lifespan,
    )
