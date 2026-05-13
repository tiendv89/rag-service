"""
Tests for services/indexer/pr_indexer.py.

Uses mock GitHub API responses and a mock Qdrant client to verify:
- Cursor advances correctly after indexing
- Correct payload fields are written to Qdrant
- Already-indexed PRs (cursor) are skipped
- Empty body is handled gracefully
- GitHub API errors are handled without crashing
"""

import json
import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.indexer.pr_indexer import PrIndexer, _point_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pr(number: int, title: str, body: str, merged_at: str, branch: str = "main") -> dict:
    return {
        "number": number,
        "title": title,
        "body": body,
        "merged_at": merged_at,
        "head": {"ref": branch},
    }


def _mock_embedder(dim: int = 4) -> MagicMock:
    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts: [[0.1] * dim for _ in texts]
    return embedder


def _mock_qdrant_client() -> MagicMock:
    client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# _point_id helper
# ---------------------------------------------------------------------------

class TestPointId:
    def test_returns_valid_uuid(self):
        pid = _point_id("rag-service", 42, 0)
        parsed = uuid.UUID(pid)
        assert str(parsed) == pid

    def test_deterministic(self):
        assert _point_id("rag-service", 42, 0) == _point_id("rag-service", 42, 0)

    def test_different_pr_numbers_differ(self):
        assert _point_id("rag-service", 1, 0) != _point_id("rag-service", 2, 0)

    def test_different_chunk_indices_differ(self):
        assert _point_id("rag-service", 1, 0) != _point_id("rag-service", 1, 1)


# ---------------------------------------------------------------------------
# PrIndexer — cursor persistence
# ---------------------------------------------------------------------------

class TestCursorPersistence:
    def test_cold_start_cursor_is_none(self, tmp_path):
        state_path = tmp_path / "pr_index_state.json"
        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )
        assert indexer._state == {}

    def test_cursor_saved_after_indexing(self, tmp_path):
        state_path = tmp_path / "pr_index_state.json"
        prs = [_make_pr(1, "Fix bug", "Details.", "2026-05-01T10:00:00Z", "feature/foo-T1")]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=prs):
            count = indexer.index_repo_prs("org/repo", "rag-service")

        assert count == 1
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["rag-service"] == "2026-05-01T10:00:00Z"

    def test_cursor_loaded_from_existing_state(self, tmp_path):
        state_path = tmp_path / "pr_index_state.json"
        state_path.write_text(json.dumps({"rag-service": "2026-04-01T00:00:00Z"}))

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )
        assert indexer._state.get("rag-service") == "2026-04-01T00:00:00Z"


# ---------------------------------------------------------------------------
# PrIndexer — payload correctness
# ---------------------------------------------------------------------------

class TestPayloadFields:
    def test_payload_has_required_fields(self, tmp_path):
        state_path = tmp_path / "state.json"
        client = _mock_qdrant_client()
        prs = [_make_pr(42, "My PR", "Body text.", "2026-05-01T10:00:00Z", "feature/agent-rag-T2")]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=client,
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=prs):
            indexer.index_repo_prs("org/repo", "rag-service")

        assert client.upsert.called
        upserted_points = client.upsert.call_args[1]["points"]
        assert len(upserted_points) >= 1

        payload = upserted_points[0].payload
        assert payload["source_type"] == "pr_description"
        assert payload["workspace_id"] == "ws"
        assert payload["repo_id"] == "rag-service"
        assert payload["pr_number"] == 42
        assert payload["pr_title"] == "My PR"
        assert payload["merged_at"] == "2026-05-01T10:00:00Z"
        assert "github.com/org/repo/pull/42" in payload["source_path"]
        assert payload["chunk_index"] == 0

    def test_feature_id_and_task_id_extracted_from_branch(self, tmp_path):
        state_path = tmp_path / "state.json"
        client = _mock_qdrant_client()
        prs = [_make_pr(10, "T", "B", "2026-05-01T10:00:00Z", "feature/agent-rag-pr-index-T3")]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=client,
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=prs):
            indexer.index_repo_prs("org/repo", "rag-service")

        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["feature_id"] == "agent-rag-pr-index"
        assert payload["task_id"] == "3"

    def test_non_feature_branch_yields_null_ids(self, tmp_path):
        state_path = tmp_path / "state.json"
        client = _mock_qdrant_client()
        prs = [_make_pr(5, "T", "B", "2026-05-01T10:00:00Z", "main")]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=client,
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=prs):
            indexer.index_repo_prs("org/repo", "rag-service")

        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["feature_id"] is None
        assert payload["task_id"] is None


# ---------------------------------------------------------------------------
# PrIndexer — cursor filtering and idempotency
# ---------------------------------------------------------------------------

class TestCursorFiltering:
    def test_no_new_prs_returns_zero(self, tmp_path):
        state_path = tmp_path / "state.json"
        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=[]):
            count = indexer.index_repo_prs("org/repo", "rag-service")

        assert count == 0

    def test_qdrant_not_called_when_no_prs(self, tmp_path):
        state_path = tmp_path / "state.json"
        client = _mock_qdrant_client()
        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=client,
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=[]):
            indexer.index_repo_prs("org/repo", "rag-service")

        client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# PrIndexer — error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_github_api_error_returns_zero(self, tmp_path):
        import urllib.error

        state_path = tmp_path / "state.json"
        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_github_get", side_effect=urllib.error.HTTPError(
            url="", code=401, msg="Unauthorized", hdrs=None, fp=None
        )):
            count = indexer.index_repo_prs("org/repo", "rag-service")

        assert count == 0

    def test_empty_pr_body_handled(self, tmp_path):
        state_path = tmp_path / "state.json"
        client = _mock_qdrant_client()
        prs = [_make_pr(1, "Title only", None, "2026-05-01T10:00:00Z")]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=client,
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_fetch_merged_prs", return_value=prs):
            count = indexer.index_repo_prs("org/repo", "rag-service")

        assert count == 1
        payload = client.upsert.call_args[1]["points"][0].payload
        assert payload["pr_title"] == "Title only"


# ---------------------------------------------------------------------------
# PrIndexer — GitHub API fetch filtering (cursor application)
# ---------------------------------------------------------------------------

class TestFetchMergedPrs:
    def test_cursor_filters_older_prs(self, tmp_path):
        """PRs with merged_at <= cursor must be skipped."""
        state_path = tmp_path / "state.json"

        raw_prs = [
            _make_pr(1, "Old PR", "body", "2026-04-01T00:00:00Z"),
            _make_pr(2, "New PR", "body", "2026-05-01T00:00:00Z"),
        ]
        page_responses = [raw_prs, []]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        call_count = 0
        def mock_github_get(url):
            nonlocal call_count
            resp = page_responses[min(call_count, len(page_responses) - 1)]
            call_count += 1
            return resp

        with patch.object(indexer, "_github_get", side_effect=mock_github_get):
            prs = indexer._fetch_merged_prs("org/repo", cursor="2026-04-15T00:00:00Z")

        assert len(prs) == 1
        assert prs[0]["number"] == 2

    def test_unmerged_prs_excluded(self, tmp_path):
        state_path = tmp_path / "state.json"
        raw_prs = [
            {"number": 1, "title": "Open PR", "body": "", "merged_at": None, "head": {"ref": "main"}},
            _make_pr(2, "Merged PR", "body", "2026-05-01T00:00:00Z"),
        ]

        indexer = PrIndexer(
            github_token="tok",
            qdrant_client=_mock_qdrant_client(),
            embedder=_mock_embedder(),
            workspace_id="ws",
            state_path=str(state_path),
        )

        with patch.object(indexer, "_github_get", side_effect=[raw_prs, []]):
            prs = indexer._fetch_merged_prs("org/repo", cursor=None)

        assert len(prs) == 1
        assert prs[0]["number"] == 2
