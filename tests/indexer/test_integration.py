"""
Integration test: index a known document → verify point retrievable in Qdrant.

This test is SKIPPED unless a live Qdrant instance is available at the URL
specified by the QDRANT_TEST_URL environment variable (default: http://localhost:6333).

Run with:
    QDRANT_TEST_URL=http://localhost:6333 pytest tests/indexer/test_integration.py -v

The test uses a unique workspace_id derived from a UUID so it does not
interfere with production collections and cleans up after itself.
"""

import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

QDRANT_TEST_URL = os.environ.get("QDRANT_TEST_URL", "")


def _qdrant_available() -> bool:
    """Return True if a Qdrant instance is reachable at QDRANT_TEST_URL."""
    if not QDRANT_TEST_URL:
        return False
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_TEST_URL, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_available(),
    reason="No live Qdrant instance at QDRANT_TEST_URL — skipping integration tests",
)


@pytest.fixture
def workspace_id() -> str:
    """Unique workspace_id per test run to avoid cross-test pollution."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_TEST_URL)


@pytest.fixture
def embedder():
    from services.indexer.embedder import Embedder

    return Embedder()


class TestIndexAndRetrieve:
    def test_index_skill_file_and_retrieve(self, tmp_path, workspace_id, qdrant_client, embedder):
        """Index a SKILL.md file and verify the point is retrievable."""
        from services.indexer.main import index_repo
        from services.indexer.git_watcher import GitWatcher
        from services.shared.qdrant_init import init_collection, query_points

        # Set up a fake skill file
        skill_dir = tmp_path / "workflow" / "workflow_skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        content = "# Test Skill\n\nThis skill verifies RAG indexing end-to-end."
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        # Initialise collection
        init_collection(qdrant_client, workspace_id)

        # Index the document
        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = [
            "workflow/workflow_skills/test-skill/SKILL.md"
        ]

        count = index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=embedder,
            client=qdrant_client,
            workspace_id=workspace_id,
        )

        assert count == 1, "Expected exactly 1 point to be upserted"

        # Query for the indexed chunk
        query_vector = embedder.encode("Test Skill RAG indexing")[0]
        results = query_points(
            qdrant_client,
            workspace_id=workspace_id,
            query_vector=query_vector,
            top_k=1,
        )

        assert len(results) == 1
        payload = results[0]["payload"]
        assert payload["workspace_id"] == workspace_id
        assert payload["source_type"] == "skill"
        assert "test-skill/SKILL.md" in payload["source_path"]
        assert results[0]["score"] > 0.0

        # Cleanup
        qdrant_client.delete_collection(workspace_id)

    def test_workspace_isolation(self, tmp_path, qdrant_client, embedder):
        """Points indexed in workspace A must not appear in workspace B queries."""
        from services.indexer.main import index_repo
        from services.indexer.git_watcher import GitWatcher
        from services.shared.qdrant_init import init_collection, query_points

        ws_a = f"test-{uuid.uuid4().hex[:8]}"
        ws_b = f"test-{uuid.uuid4().hex[:8]}"

        # Create a skill file in workspace A
        skill_dir = tmp_path / "workflow" / "workflow_skills" / "iso-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Isolation Test Skill", encoding="utf-8")

        init_collection(qdrant_client, ws_a)
        init_collection(qdrant_client, ws_b)

        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = [
            "workflow/workflow_skills/iso-skill/SKILL.md"
        ]

        index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=embedder,
            client=qdrant_client,
            workspace_id=ws_a,
        )

        # Query workspace B — should return no results
        query_vector = embedder.encode("Isolation Test Skill")[0]
        results_b = query_points(
            qdrant_client,
            workspace_id=ws_b,
            query_vector=query_vector,
            top_k=5,
        )
        assert results_b == [], "ws_b must not return points indexed into ws_a"

        # Cleanup
        qdrant_client.delete_collection(ws_a)
        qdrant_client.delete_collection(ws_b)
