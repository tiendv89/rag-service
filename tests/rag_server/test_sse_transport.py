"""
Tests for the SSE transport layer of the RAG MCP server.

Covers three things:
  1. Route structure — create_app() must expose explicit /sse and /messages/
     routes rather than a catch-all Mount("/").
  2. Stateless-session fix — the SSE handler must invoke _mcp_server.run with
     stateless=True so that a tool call arriving before the MCP 'initialized'
     notification is processed without the "before initialization was complete"
     warning.  This is the regression test for the Claude Code SSE handshake
     race that was observed in production.
  3. MCP tool reachability — rag_query must be listable and callable over the
     in-process MCP transport (tests the full tool dispatch path).
"""

import logging
from unittest.mock import MagicMock, patch

import anyio
import pytest

import services.rag_server.server as rag_server_module
from services.rag_server.server import create_app, mcp_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embedder() -> MagicMock:
    emb = MagicMock()
    emb.encode.return_value = [[0.1] * 384]
    return emb


def _make_hit(
    content: str = "chunk text",
    source_path: str = "docs/features/foo/technical-design.md",
    source_type: str = "technical_design",
    feature_id: str | None = "foo",
    score: float = 0.92,
) -> dict:
    return {
        "id": "abc123",
        "score": score,
        "payload": {
            "content": content,
            "source_path": source_path,
            "source_type": source_type,
            "feature_id": feature_id,
            "workspace_id": "workspace",
            "chunk_index": 0,
            "indexed_at": "2026-01-01T00:00:00+00:00",
        },
    }


# ---------------------------------------------------------------------------
# 1. Route structure
# ---------------------------------------------------------------------------

class TestSseRoutes:
    """create_app() must expose /sse and /messages/ as discrete named routes."""

    def test_sse_route_exists(self):
        from starlette.routing import Route

        app = create_app()
        paths = [r.path for r in app.routes if isinstance(r, Route)]
        assert "/sse" in paths

    def test_messages_mount_exists(self):
        from starlette.routing import Mount

        app = create_app()
        mount_paths = [r.path for r in app.routes if isinstance(r, Mount)]
        assert any(p.rstrip("/") == "/messages" for p in mount_paths)

    def test_no_catch_all_root_mount(self):
        """The old pattern used Mount('/') which shadowed /health and /query."""
        from starlette.routing import Mount

        app = create_app()
        root_mounts = [
            r for r in app.routes
            if isinstance(r, Mount) and r.path.rstrip("/") == ""
        ]
        assert root_mounts == []


# ---------------------------------------------------------------------------
# 2. Stateless-session fix
# ---------------------------------------------------------------------------

class TestStatelessSession:
    """
    The SSE handler must invoke _mcp_server.run with stateless=True.
    Without this flag, a tool call arriving before the MCP 'initialized'
    notification triggers a WARNING and returns an error to the client.
    """

    def setup_method(self):
        rag_server_module._state["client"] = MagicMock()
        rag_server_module._state["embedder"] = _mock_embedder()

    def teardown_method(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = None

    def test_pre_init_tool_call_no_warning(self, caplog):
        """
        Send a tool call directly to _mcp_server.run(stateless=True) without
        going through the initialize / initialized handshake first.  The
        "Received request before initialization was complete" warning must NOT
        appear, proving stateless=True suppresses the race-condition error.
        """
        import mcp.types as types
        from mcp.shared.memory import create_client_server_memory_streams
        from mcp.shared.message import SessionMessage

        async def _run():
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                client_read, client_write = client_streams
                server_read, server_write = server_streams

                async with anyio.create_task_group() as tg:
                    async def _serve():
                        await mcp_server._mcp_server.run(
                            server_read,
                            server_write,
                            mcp_server._mcp_server.create_initialization_options(),
                            stateless=True,
                        )

                    tg.start_soon(_serve)

                    raw = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "rag_query",
                            "arguments": {
                                "query": "test",
                                "workspace_id": "workspace",
                            },
                        },
                    }
                    with patch(
                        "services.rag_server.server.query_points",
                        return_value=[_make_hit()],
                    ):
                        await client_write.send(
                            SessionMessage(
                                message=types.JSONRPCMessage.model_validate(raw)
                            )
                        )
                        await client_read.receive()

                    tg.cancel_scope.cancel()

        with caplog.at_level(logging.WARNING):
            anyio.run(_run)

        init_warnings = [
            r for r in caplog.records
            if "initialization was complete" in r.message
        ]
        assert init_warnings == [], (
            f"Unexpected initialization warning(s): {[r.message for r in init_warnings]}"
        )

    def test_pre_init_tool_call_without_stateless_does_warn(self, caplog):
        """
        Counterpart: the same tool call against stateless=False (old behaviour)
        MUST produce the initialization warning.  This pins the behaviour we
        are deliberately suppressing with stateless=True.
        """
        import mcp.types as types
        from mcp.shared.memory import create_client_server_memory_streams
        from mcp.shared.message import SessionMessage

        async def _run():
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                client_read, client_write = client_streams
                server_read, server_write = server_streams

                async with anyio.create_task_group() as tg:
                    async def _serve():
                        await mcp_server._mcp_server.run(
                            server_read,
                            server_write,
                            mcp_server._mcp_server.create_initialization_options(),
                            stateless=False,  # old behaviour
                        )

                    tg.start_soon(_serve)

                    raw = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "rag_query",
                            "arguments": {
                                "query": "test",
                                "workspace_id": "workspace",
                            },
                        },
                    }
                    with patch(
                        "services.rag_server.server.query_points",
                        return_value=[_make_hit()],
                    ):
                        await client_write.send(
                            SessionMessage(
                                message=types.JSONRPCMessage.model_validate(raw)
                            )
                        )
                        await client_read.receive()

                    tg.cancel_scope.cancel()

        with caplog.at_level(logging.WARNING):
            anyio.run(_run)

        init_warnings = [
            r for r in caplog.records
            if "initialization was complete" in r.message
        ]
        assert init_warnings, "Expected initialization warning with stateless=False but none logged"


# ---------------------------------------------------------------------------
# 3. MCP tool reachability
# ---------------------------------------------------------------------------

class TestMcpToolReachability:
    """rag_query must be discoverable and callable via the MCP protocol."""

    def setup_method(self):
        rag_server_module._state["client"] = MagicMock()
        rag_server_module._state["embedder"] = _mock_embedder()

    def teardown_method(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = None

    def test_rag_query_listed_in_tools(self):
        """list_tools() must include rag_query."""
        from mcp.shared.memory import create_connected_server_and_client_session

        async def _run():
            async with create_connected_server_and_client_session(mcp_server) as client:
                return await client.list_tools()

        result = anyio.run(_run)
        tool_names = [t.name for t in result.tools]
        assert "rag_query" in tool_names

    def test_rag_query_call_returns_results(self):
        """call_tool('rag_query', ...) must return at least one content item."""
        from mcp.shared.memory import create_connected_server_and_client_session

        async def _run():
            hits = [_make_hit()]
            with patch("services.rag_server.server.query_points", return_value=hits):
                async with create_connected_server_and_client_session(mcp_server) as client:
                    return await client.call_tool(
                        "rag_query",
                        {"query": "how does indexing work", "workspace_id": "workspace"},
                    )

        result = anyio.run(_run)
        assert not result.isError
        assert len(result.content) > 0

    def test_rag_query_call_empty_results(self):
        """call_tool must succeed (not error) when Qdrant returns no hits."""
        from mcp.shared.memory import create_connected_server_and_client_session

        async def _run():
            with patch("services.rag_server.server.query_points", return_value=[]):
                async with create_connected_server_and_client_session(mcp_server) as client:
                    return await client.call_tool(
                        "rag_query",
                        {"query": "nothing matches", "workspace_id": "workspace"},
                    )

        result = anyio.run(_run)
        assert not result.isError

    def test_rag_query_call_missing_workspace_id_returns_error(self):
        """call_tool with workspace_id='' must return an MCP tool error, not crash."""
        from mcp.shared.memory import create_connected_server_and_client_session

        async def _run():
            async with create_connected_server_and_client_session(mcp_server) as client:
                return await client.call_tool(
                    "rag_query",
                    {"query": "test", "workspace_id": ""},
                )

        result = anyio.run(_run)
        assert result.isError
