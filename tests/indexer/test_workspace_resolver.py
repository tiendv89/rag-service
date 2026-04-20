"""
Unit tests for services/indexer/workspace_resolver.py.
"""

import os
from pathlib import Path

import pytest
import yaml

from services.indexer.workspace_resolver import load_repo_paths


def _write_workspace_yaml(path: Path, repos: list[dict]) -> str:
    """Write a minimal workspace.yaml and return its path string."""
    config = {"workspace_id": "test-ws", "repos": repos}
    yaml_path = path / "workspace.yaml"
    yaml_path.write_text(yaml.dump(config), encoding="utf-8")
    return str(yaml_path)


class TestLoadRepoPaths:
    def test_literal_paths_returned_as_is(self, tmp_path):
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "repo-a", "local_path": "/repos/a"},
                {"id": "repo-b", "local_path": "/repos/b"},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == ["/repos/a", "/repos/b"]

    def test_env_reference_resolved_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_REPO_PATH", "/container/my-repo")
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "my-repo", "local_path": "env:MY_REPO_PATH"}],
        )
        result = load_repo_paths(yaml_path)
        assert result == ["/container/my-repo"]

    def test_unset_env_reference_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "present", "local_path": "/repos/present"},
                {"id": "missing", "local_path": "env:MISSING_VAR"},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == ["/repos/present"]

    def test_repo_without_local_path_skipped(self, tmp_path):
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "no-path"},
                {"id": "with-path", "local_path": "/repos/present"},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == ["/repos/present"]

    def test_missing_workspace_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_repo_paths(str(tmp_path / "nonexistent.yaml"))

    def test_empty_repos_list_raises(self, tmp_path):
        yaml_path = _write_workspace_yaml(tmp_path, [])
        with pytest.raises(ValueError, match="no repos\\[\\] entries"):
            load_repo_paths(yaml_path)

    def test_all_repos_unresolvable_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("UNSET_A", raising=False)
        monkeypatch.delenv("UNSET_B", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "a", "local_path": "env:UNSET_A"},
                {"id": "b", "local_path": "env:UNSET_B"},
            ],
        )
        # Returns empty list (caller decides if that's an error)
        result = load_repo_paths(yaml_path)
        assert result == []

    def test_mixed_literal_and_env_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WF_PATH", "/repos/workflow")
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "mgmt", "local_path": "/repos/management"},
                {"id": "workflow", "local_path": "env:WF_PATH"},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == ["/repos/management", "/repos/workflow"]

    def test_empty_local_path_string_skipped(self, tmp_path):
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "empty-path", "local_path": ""},
                {"id": "ok", "local_path": "/repos/ok"},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == ["/repos/ok"]
