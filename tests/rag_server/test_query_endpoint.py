"""
Unit and integration tests for the POST /query REST endpoint on rag-server.

Unit tests do not require a running Qdrant instance — they verify request
validation, response shaping, and error handling via mocks.

Integration tests require Qdrant and are auto-skipped when unavailable.
"""

import uuid
import os
import pytest
from unittest.mock import patch, MagicMock

from starlette.testclient import TestClient

import services.rag_server.server as rag_server_module
from services.rag_server.server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embedder(vector: list[float] | None = None) -> MagicMock:
    emb = MagicMock()
    emb.encode.return_value = [vector or [0.1] * 768]
    return emb


def _mock_client() -> MagicMock:
    return MagicMock()


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
# Fixture: minimal Starlette app wrapping only the query_endpoint handler
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    TestClient wired to the full app with mocked Qdrant state.

    We use startup_retries=1 and a non-routable URL so Qdrant connection fails
    fast. After the lifespan runs, we inject mock state so _rag_query works.
    """
    rag_server_module._state["client"] = _mock_client()
    rag_server_module._state["embedder"] = _mock_embedder()

    app = create_app(qdrant_url="http://127.0.0.1:19999", startup_retries=1, startup_retry_delay=0.0)

    with TestClient(app, raise_server_exceptions=False) as c:
        # Lifespan may have reset state on Qdrant connection failure — re-inject.
        rag_server_module._state["client"] = _mock_client()
        rag_server_module._state["embedder"] = _mock_embedder()
        yield c

    rag_server_module._state["client"] = None
    rag_server_module._state["embedder"] = None


# ---------------------------------------------------------------------------
# Unit tests — valid payload → 200 with results
# ---------------------------------------------------------------------------

class TestQueryEndpointValidRequest:
    """POST /query with valid payload must return 200 and a results list."""

    def test_returns_200_with_results(self, client):
        hits = [_make_hit()]
        with patch("services.rag_server.server.query_points", return_value=hits):
            resp = client.post("/query", json={
                "query": "how does authentication work",
                "workspace_id": "workspace",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) == 1

    def test_result_shape(self, client):
        """Each result must have content, score, and metadata with required keys."""
        with patch("services.rag_server.server.query_points", return_value=[_make_hit()]):
            resp = client.post("/query", json={
                "query": "RAG design",
                "workspace_id": "workspace",
            })
        assert resp.status_code == 200
        r = resp.json()["results"][0]
        assert "content" in r
        assert "score" in r
        assert "metadata" in r
        meta = r["metadata"]
        assert "source_type" in meta
        assert "file_path" in meta
        assert "feature_id" in meta

    def test_result_values(self, client):
        """Result values must match the indexed chunk."""
        hit = _make_hit(
            content="The indexer polls every 5 minutes.",
            source_path="docs/features/agent-rag-mcp/technical-design.md",
            source_type="technical_design",
            feature_id="agent-rag-mcp",
            score=0.88,
        )
        with patch("services.rag_server.server.query_points", return_value=[hit]):
            resp = client.post("/query", json={
                "query": "indexer polling",
                "workspace_id": "workspace",
            })
        assert resp.status_code == 200
        r = resp.json()["results"][0]
        assert r["content"] == "The indexer polls every 5 minutes."
        assert r["score"] == pytest.approx(0.88)
        assert r["metadata"]["source_type"] == "technical_design"
        assert r["metadata"]["file_path"] == "docs/features/agent-rag-mcp/technical-design.md"
        assert r["metadata"]["feature_id"] == "agent-rag-mcp"

    def test_empty_results_returns_200(self, client):
        """Unknown workspace_id must return 200 with empty results, not an error."""
        with patch("services.rag_server.server.query_points", return_value=[]):
            resp = client.post("/query", json={
                "query": "anything",
                "workspace_id": "nonexistent-workspace",
            })
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_top_k_forwarded(self, client):
        """top_k parameter must be forwarded to the retrieval layer."""
        hits = [_make_hit(score=0.9 - i * 0.1) for i in range(3)]
        with patch("services.rag_server.server.query_points", return_value=hits) as mock_qp:
            client.post("/query", json={
                "query": "test",
                "workspace_id": "workspace",
                "top_k": 3,
            })
        _, kwargs = mock_qp.call_args
        assert kwargs["top_k"] == 3

    def test_source_types_forwarded(self, client):
        """source_types must be forwarded to the retrieval layer."""
        with patch("services.rag_server.server.query_points", return_value=[]) as mock_qp:
            client.post("/query", json={
                "query": "skill context",
                "workspace_id": "workspace",
                "source_types": ["skill", "doc"],
            })
        _, kwargs = mock_qp.call_args
        assert kwargs["source_types"] == ["skill", "doc"]

    def test_source_types_omitted_means_all(self, client):
        """Omitting source_types must pass None to the retrieval layer (= all types)."""
        with patch("services.rag_server.server.query_points", return_value=[]) as mock_qp:
            client.post("/query", json={
                "query": "test",
                "workspace_id": "workspace",
            })
        _, kwargs = mock_qp.call_args
        assert kwargs["source_types"] is None

    def test_null_feature_id_preserved(self, client):
        """feature_id=None must be returned as null in the response."""
        hit = _make_hit(feature_id=None)
        with patch("services.rag_server.server.query_points", return_value=[hit]):
            resp = client.post("/query", json={
                "query": "test",
                "workspace_id": "workspace",
            })
        assert resp.json()["results"][0]["metadata"]["feature_id"] is None

    def test_multiple_results_returned(self, client):
        """All hits returned by Qdrant must appear in the response."""
        hits = [_make_hit(score=0.9 - i * 0.1) for i in range(3)]
        with patch("services.rag_server.server.query_points", return_value=hits):
            resp = client.post("/query", json={
                "query": "test",
                "workspace_id": "workspace",
                "top_k": 3,
            })
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 3


# ---------------------------------------------------------------------------
# Unit tests — missing required fields → 422
# ---------------------------------------------------------------------------

class TestQueryEndpointValidation:
    """POST /query with malformed or missing required fields must return 422."""

    def test_missing_query_field(self, client):
        resp = client.post("/query", json={"workspace_id": "workspace"})
        assert resp.status_code == 422

    def test_missing_workspace_id_field(self, client):
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 422

    def test_empty_body(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_non_json_body(self, client):
        resp = client.post(
            "/query",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Unit tests — Qdrant error → 500
# ---------------------------------------------------------------------------

class TestQueryEndpoint500:
    """POST /query must return 500 with {\"error\": \"...\"} when Qdrant raises."""

    def test_qdrant_error_returns_500(self, client):
        with patch(
            "services.rag_server.server.query_points",
            side_effect=RuntimeError("Qdrant down"),
        ):
            resp = client.post("/query", json={
                "query": "test",
                "workspace_id": "workspace",
            })
        assert resp.status_code == 500
        body = resp.json()
        assert "error" in body
        assert "Qdrant down" in body["error"]

    def test_500_error_field_is_string(self, client):
        with patch(
            "services.rag_server.server.query_points",
            side_effect=Exception("unexpected"),
        ):
            resp = client.post("/query", json={
                "query": "test",
                "workspace_id": "workspace",
            })
        assert resp.status_code == 500
        assert isinstance(resp.json()["error"], str)


# ---------------------------------------------------------------------------
# Integration tests — requires live Qdrant
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
TEST_WORKSPACE = "test-integration-query-endpoint"


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url=QDRANT_URL).get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason=f"Qdrant not reachable at {QDRANT_URL}",
)


@pytest.fixture(scope="module")
def live_client():
    """TestClient wired to a real Qdrant instance with one seeded doc."""
    from qdrant_client import QdrantClient
    from services.rag_server.embedder import Embedder
    from services.shared.qdrant_init import init_collection, upsert_points

    qdrant = QdrantClient(url=QDRANT_URL)
    init_collection(qdrant, TEST_WORKSPACE)

    embedder = Embedder()
    doc_text = (
        "The POST /query endpoint wraps the internal _rag_query function "
        "and returns ranked chunks as plain JSON without SSE."
    )
    vectors = embedder.encode(doc_text)
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{TEST_WORKSPACE}|query-ep|0"))
    payload = {
        "workspace_id": TEST_WORKSPACE,
        "source_type": "technical_design",
        "source_path": "docs/features/agent-rag-v2/technical-design.md",
        "feature_id": "agent-rag-v2",
        "chunk_index": 0,
        "indexed_at": "2026-04-22T00:00:00+00:00",
        "content": doc_text,
    }
    upsert_points(
        qdrant,
        TEST_WORKSPACE,
        [{"id": point_id, "vector": vectors[0], "payload": payload}],
    )

    rag_server_module._state["client"] = qdrant
    rag_server_module._state["embedder"] = embedder

    app = create_app(qdrant_url=QDRANT_URL, startup_retries=1, startup_retry_delay=0.0)
    with TestClient(app, raise_server_exceptions=False) as c:
        rag_server_module._state["client"] = qdrant
        rag_server_module._state["embedder"] = embedder
        yield c

    try:
        qdrant.delete_collection(TEST_WORKSPACE)
    except Exception:
        pass
    rag_server_module._state["client"] = None
    rag_server_module._state["embedder"] = None


@requires_qdrant
class TestQueryEndpointIntegration:
    """End-to-end: seed Qdrant → POST /query → verify result appears."""

    def test_indexed_doc_returned(self, live_client):
        resp = live_client.post("/query", json={
            "query": "POST query endpoint plain JSON ranked chunks",
            "workspace_id": TEST_WORKSPACE,
        })
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1
        assert results[0]["score"] > 0
        assert results[0]["metadata"]["source_type"] == "technical_design"

    def test_content_field_present(self, live_client):
        resp = live_client.post("/query", json={
            "query": "_rag_query function SSE endpoint",
            "workspace_id": TEST_WORKSPACE,
        })
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1
        assert "POST /query" in results[0]["content"]

    def test_unknown_workspace_returns_empty(self, live_client):
        resp = live_client.post("/query", json={
            "query": "anything",
            "workspace_id": "completely-different-workspace-xyz",
        })
        assert resp.status_code == 200
        assert resp.json()["results"] == []
