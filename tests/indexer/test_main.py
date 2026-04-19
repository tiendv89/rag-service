"""
Unit tests for services/indexer/main.py.

These tests cover the core indexing logic without requiring a live
Qdrant instance or real git repositories.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.indexer.main import _point_id, index_repo
from services.indexer.git_watcher import GitWatcher
from services.indexer.embedder import Embedder


# ---------------------------------------------------------------------------
# _point_id
# ---------------------------------------------------------------------------

class TestPointId:
    def test_returns_string(self):
        pid = _point_id("/repos/workspace", "README.md", 0)
        assert isinstance(pid, str)

    def test_deterministic(self):
        a = _point_id("/repos/workspace", "README.md", 0)
        b = _point_id("/repos/workspace", "README.md", 0)
        assert a == b

    def test_different_chunk_index_produces_different_id(self):
        a = _point_id("/repos/workspace", "README.md", 0)
        b = _point_id("/repos/workspace", "README.md", 1)
        assert a != b

    def test_different_paths_produce_different_ids(self):
        a = _point_id("/repos/workspace", "README.md", 0)
        b = _point_id("/repos/workspace", "CLAUDE.md", 0)
        assert a != b

    def test_id_length_is_32(self):
        pid = _point_id("/repos/workspace", "README.md", 0)
        assert len(pid) == 32


# ---------------------------------------------------------------------------
# index_repo
# ---------------------------------------------------------------------------

class TestIndexRepo:
    def _make_embedder(self, dim: int = 384) -> MagicMock:
        embedder = MagicMock(spec=Embedder)
        embedder.encode.side_effect = lambda texts: [[0.1] * dim for _ in texts]
        return embedder

    def test_no_changed_files_returns_zero(self, tmp_path):
        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = []
        client = MagicMock()

        result = index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=self._make_embedder(),
            client=client,
            workspace_id="workspace",
        )

        assert result == 0
        client.upsert.assert_not_called()

    def test_indexes_skill_file(self, tmp_path):
        # Create a fake SKILL.md
        skill_dir = tmp_path / "workflow" / "workflow_skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# My Skill\nDoes something useful.", encoding="utf-8")

        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = [
            "workflow/workflow_skills/my-skill/SKILL.md"
        ]
        client = MagicMock()
        embedder = self._make_embedder()

        result = index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=embedder,
            client=client,
            workspace_id="workspace",
        )

        # Skill is whole-file → 1 chunk → 1 point
        assert result == 1
        client.upsert.assert_called_once()
        upsert_kwargs = client.upsert.call_args[1]
        assert upsert_kwargs["collection_name"] == "workspace"
        points = upsert_kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["workspace_id"] == "workspace"
        assert points[0].payload["source_type"] == "skill"

    def test_skips_unclassified_file(self, tmp_path):
        # Create a source code file that should not be indexed
        src = tmp_path / "services" / "shared"
        src.mkdir(parents=True)
        (src / "schema.py").write_text("class Foo: pass", encoding="utf-8")

        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = ["services/shared/schema.py"]
        client = MagicMock()

        result = index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=self._make_embedder(),
            client=client,
            workspace_id="workspace",
        )

        assert result == 0
        client.upsert.assert_not_called()

    def test_skips_deleted_file(self, tmp_path):
        # Report a changed file that doesn't exist on disk
        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = ["README.md"]
        # Don't create the file — it is "deleted"
        client = MagicMock()

        result = index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=self._make_embedder(),
            client=client,
            workspace_id="workspace",
        )

        assert result == 0

    def test_indexes_readme_into_multiple_chunks(self, tmp_path):
        # Create a large README that will be chunked
        readme = tmp_path / "README.md"
        readme.write_text("word " * 600, encoding="utf-8")  # ~3000 chars

        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = ["README.md"]
        client = MagicMock()
        embedder = self._make_embedder()

        result = index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=embedder,
            client=client,
            workspace_id="workspace",
        )

        # Large README → multiple chunks → result > 1
        assert result > 1

    def test_point_payload_has_workspace_id(self, tmp_path):
        skill_dir = tmp_path / "workflow" / "workflow_skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test", encoding="utf-8")

        watcher = MagicMock(spec=GitWatcher)
        watcher.changed_files.return_value = [
            "workflow/workflow_skills/test-skill/SKILL.md"
        ]
        client = MagicMock()

        index_repo(
            repo_path=str(tmp_path),
            watcher=watcher,
            embedder=self._make_embedder(),
            client=client,
            workspace_id="myworkspace",
        )

        upsert_kwargs = client.upsert.call_args[1]
        points = upsert_kwargs["points"]
        for p in points:
            assert p.payload["workspace_id"] == "myworkspace"
