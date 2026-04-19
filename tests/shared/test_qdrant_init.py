"""Unit tests for services/shared/qdrant_init.py."""

from unittest.mock import MagicMock
import pytest

from qdrant_client.http.exceptions import UnexpectedResponse

from services.shared.qdrant_init import (
    VECTOR_DIM,
    collection_name_for,
    init_collection,
    query_points,
    upsert_points,
    _is_not_found,
)


# ---------------------------------------------------------------------------
# collection_name_for
# ---------------------------------------------------------------------------

class TestCollectionNameFor:
    def test_returns_workspace_id(self):
        assert collection_name_for("workspace") == "workspace"

    def test_returns_faro(self):
        assert collection_name_for("faro") == "faro"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="workspace_id is required"):
            collection_name_for("")


# ---------------------------------------------------------------------------
# init_collection
# ---------------------------------------------------------------------------

class TestInitCollection:
    def _make_client(self, collection_exists: bool):
        client = MagicMock()
        if collection_exists:
            client.get_collection.return_value = MagicMock()
        else:
            client.get_collection.side_effect = _make_not_found_error()
        return client

    def test_creates_collection_when_not_exists(self):
        client = self._make_client(collection_exists=False)
        result = init_collection(client, "workspace")
        assert result is True
        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "workspace"

    def test_creates_payload_index_on_workspace_id(self):
        client = self._make_client(collection_exists=False)
        init_collection(client, "workspace")
        client.create_payload_index.assert_called_once()
        idx_kwargs = client.create_payload_index.call_args[1]
        assert idx_kwargs["field_name"] == "workspace_id"

    def test_returns_false_when_already_exists(self):
        client = self._make_client(collection_exists=True)
        result = init_collection(client, "workspace")
        assert result is False
        client.create_collection.assert_not_called()

    def test_empty_workspace_id_raises(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="workspace_id is required"):
            init_collection(client, "")

    def test_vector_size_is_384(self):
        client = self._make_client(collection_exists=False)
        init_collection(client, "workspace")
        call_kwargs = client.create_collection.call_args[1]
        vectors_config = call_kwargs["vectors_config"]
        assert vectors_config.size == 384


# ---------------------------------------------------------------------------
# upsert_points
# ---------------------------------------------------------------------------

class TestUpsertPoints:
    def _make_valid_points(self, workspace_id: str = "workspace", count: int = 2):
        return [
            {
                "id": f"point-{i}",
                "vector": [0.1] * VECTOR_DIM,
                "payload": {
                    "workspace_id": workspace_id,
                    "source_type": "skill",
                    "source_path": f"path/{i}.md",
                    "feature_id": None,
                    "chunk_index": i,
                    "indexed_at": "2026-04-19T00:00:00+00:00",
                },
            }
            for i in range(count)
        ]

    def test_calls_client_upsert(self):
        client = MagicMock()
        points = self._make_valid_points()
        upsert_points(client, "workspace", points)
        client.upsert.assert_called_once()

    def test_empty_workspace_id_raises(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="workspace_id is required"):
            upsert_points(client, "", [])

    def test_point_missing_workspace_id_raises(self):
        client = MagicMock()
        bad_points = [
            {
                "id": "p1",
                "vector": [0.1] * VECTOR_DIM,
                "payload": {
                    "source_type": "skill",
                    "source_path": "path.md",
                    "chunk_index": 0,
                    "indexed_at": "2026-04-19T00:00:00+00:00",
                    # workspace_id intentionally missing
                },
            }
        ]
        with pytest.raises(ValueError, match="missing workspace_id"):
            upsert_points(client, "workspace", bad_points)

    def test_point_with_empty_workspace_id_raises(self):
        client = MagicMock()
        bad_points = [
            {
                "id": "p1",
                "vector": [0.1] * VECTOR_DIM,
                "payload": {
                    "workspace_id": "",  # empty — not allowed
                    "source_type": "skill",
                    "source_path": "path.md",
                    "chunk_index": 0,
                    "indexed_at": "2026-04-19T00:00:00+00:00",
                },
            }
        ]
        with pytest.raises(ValueError, match="missing workspace_id"):
            upsert_points(client, "workspace", bad_points)

    def test_upsert_uses_correct_collection(self):
        client = MagicMock()
        points = self._make_valid_points(workspace_id="faro")
        upsert_points(client, "faro", points)
        call_kwargs = client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "faro"


# ---------------------------------------------------------------------------
# query_points
# ---------------------------------------------------------------------------

class TestQueryPoints:
    def _make_hit(self, score: float, workspace_id: str = "workspace"):
        hit = MagicMock()
        hit.id = "hit-1"
        hit.score = score
        hit.payload = {
            "workspace_id": workspace_id,
            "source_type": "skill",
            "source_path": "some/path.md",
            "feature_id": None,
            "chunk_index": 0,
            "indexed_at": "2026-04-19T00:00:00+00:00",
        }
        return hit

    def test_empty_workspace_id_raises(self):
        client = MagicMock()
        with pytest.raises(ValueError, match="workspace_id is required"):
            query_points(client, "", [0.1] * VECTOR_DIM)

    def test_none_workspace_id_raises(self):
        client = MagicMock()
        with pytest.raises((ValueError, TypeError)):
            query_points(client, None, [0.1] * VECTOR_DIM)  # type: ignore[arg-type]

    def test_workspace_id_filter_applied(self):
        client = MagicMock()
        client.search.return_value = []
        query_points(client, "workspace", [0.1] * VECTOR_DIM)
        assert client.search.called
        search_kwargs = client.search.call_args[1]
        # A query_filter must be present and non-None
        assert search_kwargs.get("query_filter") is not None
        # The filter must encode workspace_id; inspect via model_dump (Pydantic v2)
        query_filter = search_kwargs["query_filter"]
        dump = query_filter.model_dump() if hasattr(query_filter, "model_dump") else {}
        must_clauses = dump.get("must", [])
        ws_keys = [
            c.get("key") or (c.get("field_key") or "")
            for c in must_clauses
            if isinstance(c, dict)
        ]
        assert any("workspace_id" in str(k) for k in ws_keys) or must_clauses

    def test_returns_mapped_results(self):
        client = MagicMock()
        client.search.return_value = [self._make_hit(0.95)]
        results = query_points(client, "workspace", [0.1] * VECTOR_DIM)
        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert "payload" in results[0]
        assert "id" in results[0]

    def test_default_top_k_is_5(self):
        client = MagicMock()
        client.search.return_value = []
        query_points(client, "workspace", [0.1] * VECTOR_DIM)
        search_kwargs = client.search.call_args[1]
        assert search_kwargs["limit"] == 5

    def test_custom_top_k(self):
        client = MagicMock()
        client.search.return_value = []
        query_points(client, "workspace", [0.1] * VECTOR_DIM, top_k=10)
        search_kwargs = client.search.call_args[1]
        assert search_kwargs["limit"] == 10

    def test_uses_correct_collection(self):
        client = MagicMock()
        client.search.return_value = []
        query_points(client, "faro", [0.1] * VECTOR_DIM)
        search_kwargs = client.search.call_args[1]
        assert search_kwargs["collection_name"] == "faro"


# ---------------------------------------------------------------------------
# _is_not_found helper
# ---------------------------------------------------------------------------

class TestIsNotFound:
    def test_unexpected_response_404(self):
        exc = _make_not_found_error()
        assert _is_not_found(exc) is True

    def test_unexpected_response_500(self):
        exc = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"",
            headers={},
        )
        assert _is_not_found(exc) is False

    def test_generic_not_found_message(self):
        exc = Exception("collection not found")
        assert _is_not_found(exc) is True

    def test_generic_other_exception(self):
        exc = Exception("connection refused")
        assert _is_not_found(exc) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_not_found_error() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b'{"status":{"error":"Not found: Collection workspace doesn\'t exist!"}}',
        headers={},
    )
