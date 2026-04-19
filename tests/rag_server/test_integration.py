"""
Integration tests for the RAG MCP server.

These tests require a running Qdrant instance (QDRANT_URL env var).
They are automatically skipped when Qdrant is not reachable.

Test plan:
  1. Index a synthetic document directly into Qdrant via the shared
     upsert_points helper (mimicking what the T2 indexer would do).
  2. Call rag_query and verify the indexed chunk is returned.
  3. Call rag_query with no workspace_id — confirm rejection.
"""

import os
import uuid
import pytest

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from services.rag_server.embedder import Embedder
from services.rag_server.server import rag_query
import services.rag_server.server as rag_server_module
from services.shared.qdrant_init import init_collection, upsert_points


# ---------------------------------------------------------------------------
# Fixtures and skip logic
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
TEST_WORKSPACE = "test-integration-rag-server"


def _qdrant_reachable() -> bool:
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason=f"Qdrant not reachable at {QDRANT_URL}",
)


@pytest.fixture(scope="module")
def qdrant_client():
    client = QdrantClient(url=QDRANT_URL)
    yield client
    # Cleanup: delete the test collection after the module finishes
    try:
        client.delete_collection(TEST_WORKSPACE)
    except Exception:
        pass


@pytest.fixture(scope="module", autouse=True)
def seed_index(qdrant_client):
    """Create a collection and seed one document chunk so rag_query has data."""
    if not _qdrant_reachable():
        yield
        return

    init_collection(qdrant_client, TEST_WORKSPACE)

    embedder = Embedder()

    doc_text = (
        "The RAG MCP server embeds the incoming query with sentence-transformers "
        "and searches Qdrant filtered by workspace_id, returning ranked chunks."
    )
    vectors = embedder.encode(doc_text)

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{TEST_WORKSPACE}|test-doc|0"))

    payload = {
        "workspace_id": TEST_WORKSPACE,
        "source_type": "technical_design",
        "source_path": "docs/features/agent-rag-mcp/technical-design.md",
        "feature_id": "agent-rag-mcp",
        "chunk_index": 0,
        "indexed_at": "2026-04-19T00:00:00+00:00",
        "content": doc_text,
    }

    upsert_points(
        qdrant_client,
        TEST_WORKSPACE,
        [{"id": point_id, "vector": vectors[0], "payload": payload}],
    )

    # Wire state for rag_query
    rag_server_module._state["client"] = qdrant_client
    rag_server_module._state["embedder"] = embedder

    yield

    rag_server_module._state["client"] = None
    rag_server_module._state["embedder"] = None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

import asyncio


def _run(coro):
    return asyncio.run(coro)


@requires_qdrant
class TestRagQueryIntegration:
    """End-to-end: index → query → verify chunk returned."""

    def test_indexed_chunk_is_returned(self):
        """A doc indexed into Qdrant must appear in rag_query results."""
        results = _run(rag_query(
            query="RAG server Qdrant workspace_id filter",
            workspace_id=TEST_WORKSPACE,
            top_k=5,
        ))
        assert len(results) >= 1
        top = results[0]
        assert top["source_type"] == "technical_design"
        assert top["source_path"] == "docs/features/agent-rag-mcp/technical-design.md"
        assert top["feature_id"] == "agent-rag-mcp"
        assert isinstance(top["score"], float)
        assert top["score"] > 0

    def test_content_field_returned(self):
        """content field must be the stored chunk text."""
        results = _run(rag_query(
            query="sentence-transformers embedding",
            workspace_id=TEST_WORKSPACE,
            top_k=1,
        ))
        assert len(results) >= 1
        assert "sentence-transformers" in results[0]["content"]

    def test_source_type_filter_returns_correct_type(self):
        """source_types filter must restrict results to the requested type."""
        results = _run(rag_query(
            query="RAG server",
            workspace_id=TEST_WORKSPACE,
            source_types=["technical_design"],
        ))
        for r in results:
            assert r["source_type"] == "technical_design"

    def test_source_type_filter_excludes_other_types(self):
        """Filtering by skill must return no results (we only indexed technical_design)."""
        results = _run(rag_query(
            query="RAG server",
            workspace_id=TEST_WORKSPACE,
            source_types=["skill"],
        ))
        assert results == []

    def test_wrong_workspace_returns_no_results(self):
        """A different workspace_id must not see chunks from TEST_WORKSPACE."""
        results = _run(rag_query(
            query="RAG server",
            workspace_id="completely-different-workspace",
        ))
        # The other workspace's collection likely doesn't exist — the query
        # layer will raise because the collection is missing; catch that.
        # If it does exist and is empty, we get [].
        assert results == []

    def test_no_workspace_id_rejected(self):
        """workspace_id='' must raise ValueError (smoke test against real Qdrant)."""
        with pytest.raises(ValueError, match="workspace_id is required"):
            _run(rag_query(query="anything", workspace_id=""))
