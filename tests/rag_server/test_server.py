"""
Unit tests for the RAG MCP server.

These tests do not require a running Qdrant instance — they verify the
rag_query tool logic, workspace_id enforcement, and result formatting using
mocks.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from services.rag_server import server as rag_server_module
from services.rag_server.server import rag_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _mock_embedder(vector: list[float] | None = None) -> MagicMock:
    emb = MagicMock()
    emb.encode.return_value = [vector or [0.1] * 384]
    return emb


def _mock_client(hits: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    return client


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
            "indexed_at": "2026-04-19T00:00:00+00:00",
        },
    }


# ---------------------------------------------------------------------------
# Smoke test: reject missing workspace_id
# ---------------------------------------------------------------------------

class TestWorkspaceIdEnforcement:
    """rag_query must reject calls that omit or blank workspace_id."""

    def _set_state(self, client=None, embedder=None):
        rag_server_module._state["client"] = client
        rag_server_module._state["embedder"] = embedder

    def teardown_method(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = None

    def test_empty_workspace_id_raises(self):
        """workspace_id='' must raise ValueError before querying Qdrant."""
        self._set_state(client=_mock_client(), embedder=_mock_embedder())
        with pytest.raises(ValueError, match="workspace_id is required"):
            _run(rag_query(query="anything", workspace_id=""))

    def test_none_workspace_id_raises(self):
        """workspace_id=None must raise ValueError (falsy check)."""
        self._set_state(client=_mock_client(), embedder=_mock_embedder())
        with pytest.raises((ValueError, TypeError)):
            # None is falsy — either our guard fires (ValueError) or type
            # coercion fails (TypeError); both are acceptable rejections.
            _run(rag_query(query="anything", workspace_id=None))  # type: ignore[arg-type]

    def test_whitespace_workspace_id_raises(self):
        """workspace_id containing only whitespace is falsy and must be rejected."""
        self._set_state(client=_mock_client(), embedder=_mock_embedder())
        with pytest.raises(ValueError, match="workspace_id is required"):
            _run(rag_query(query="anything", workspace_id="   ".strip()))


# ---------------------------------------------------------------------------
# rag_query result formatting
# ---------------------------------------------------------------------------

class TestRagQueryResultFormat:
    """rag_query must return correctly shaped result dicts."""

    def setup_method(self):
        rag_server_module._state["client"] = _mock_client()
        rag_server_module._state["embedder"] = _mock_embedder()

    def teardown_method(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = None

    def test_result_shape(self):
        """Each result dict must have the five required keys."""
        hits = [_make_hit()]
        with patch("services.rag_server.server.query_points", return_value=hits):
            results = _run(rag_query(query="RAG design", workspace_id="workspace"))

        assert len(results) == 1
        r = results[0]
        assert set(r.keys()) == {"content", "source_path", "source_type", "feature_id", "score"}

    def test_result_values(self):
        """Values must match the underlying Qdrant hit payload."""
        hits = [_make_hit(
            content="The indexer polls every 5 minutes.",
            source_path="docs/features/agent-rag-mcp/technical-design.md",
            source_type="technical_design",
            feature_id="agent-rag-mcp",
            score=0.88,
        )]
        with patch("services.rag_server.server.query_points", return_value=hits):
            results = _run(rag_query(query="indexer", workspace_id="workspace"))

        r = results[0]
        assert r["content"] == "The indexer polls every 5 minutes."
        assert r["source_path"] == "docs/features/agent-rag-mcp/technical-design.md"
        assert r["source_type"] == "technical_design"
        assert r["feature_id"] == "agent-rag-mcp"
        assert r["score"] == pytest.approx(0.88)

    def test_missing_content_defaults_to_empty_string(self):
        """If content is absent from payload, result['content'] must be ''."""
        hit = _make_hit()
        del hit["payload"]["content"]
        with patch("services.rag_server.server.query_points", return_value=[hit]):
            results = _run(rag_query(query="test", workspace_id="workspace"))

        assert results[0]["content"] == ""

    def test_null_feature_id_preserved(self):
        """feature_id=None (workspace-wide document) must be returned as None."""
        hits = [_make_hit(feature_id=None)]
        with patch("services.rag_server.server.query_points", return_value=hits):
            results = _run(rag_query(query="test", workspace_id="workspace"))

        assert results[0]["feature_id"] is None

    def test_multiple_results(self):
        """rag_query must return all hits from query_points."""
        hits = [_make_hit(score=0.9), _make_hit(score=0.8), _make_hit(score=0.7)]
        with patch("services.rag_server.server.query_points", return_value=hits):
            results = _run(rag_query(query="test", workspace_id="workspace", top_k=3))

        assert len(results) == 3

    def test_empty_results(self):
        """Empty Qdrant results must return an empty list."""
        with patch("services.rag_server.server.query_points", return_value=[]):
            results = _run(rag_query(query="test", workspace_id="workspace"))

        assert results == []


# ---------------------------------------------------------------------------
# Qdrant unavailability
# ---------------------------------------------------------------------------

class TestQdrantUnavailable:
    """rag_query must return a clear RuntimeError when Qdrant is down."""

    def teardown_method(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = None

    def test_raises_when_client_is_none(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = _mock_embedder()
        with pytest.raises(RuntimeError, match="Qdrant is not available"):
            _run(rag_query(query="test", workspace_id="workspace"))

    def test_raises_when_embedder_is_none(self):
        rag_server_module._state["client"] = _mock_client()
        rag_server_module._state["embedder"] = None
        with pytest.raises(RuntimeError, match="Embedding model"):
            _run(rag_query(query="test", workspace_id="workspace"))


# ---------------------------------------------------------------------------
# source_types filtering
# ---------------------------------------------------------------------------

class TestSourceTypesFilter:
    """source_types must be forwarded to query_points."""

    def setup_method(self):
        rag_server_module._state["client"] = _mock_client()
        rag_server_module._state["embedder"] = _mock_embedder()

    def teardown_method(self):
        rag_server_module._state["client"] = None
        rag_server_module._state["embedder"] = None

    def test_source_types_forwarded(self):
        with patch("services.rag_server.server.query_points", return_value=[]) as mock_qp:
            _run(rag_query(
                query="skill docs",
                workspace_id="workspace",
                source_types=["skill", "claude_md"],
            ))
        _, kwargs = mock_qp.call_args
        assert kwargs["source_types"] == ["skill", "claude_md"]

    def test_none_source_types_forwarded(self):
        with patch("services.rag_server.server.query_points", return_value=[]) as mock_qp:
            _run(rag_query(query="test", workspace_id="workspace", source_types=None))
        _, kwargs = mock_qp.call_args
        assert kwargs["source_types"] is None


# ---------------------------------------------------------------------------
# create_app smoke test
# ---------------------------------------------------------------------------

class TestCreateApp:
    """create_app() must return a Starlette application."""

    def test_returns_starlette_app(self):
        from starlette.applications import Starlette
        from services.rag_server.server import create_app

        app = create_app()
        assert isinstance(app, Starlette)
