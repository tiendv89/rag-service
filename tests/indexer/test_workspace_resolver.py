"""
Unit tests for services/indexer/workspace_resolver.py.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from services.indexer.workspace_resolver import (
    _build_git_ssh_command,
    _ensure_cloned,
    _resolve_ssh_key,
    load_repo_paths,
)


def _write_workspace_yaml(path: Path, repos: list[dict]) -> str:
    """Write a minimal workspace.yaml and return its path string."""
    config = {"workspace_id": "test-ws", "repos": repos}
    yaml_path = path / "workspace.yaml"
    yaml_path.write_text(yaml.dump(config), encoding="utf-8")
    return str(yaml_path)


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

class TestBuildGitSshCommand:
    def test_returns_ssh_command_string(self):
        cmd = _build_git_ssh_command("/home/user/.ssh/id_rsa")
        assert "ssh" in cmd
        assert "/home/user/.ssh/id_rsa" in cmd
        assert "StrictHostKeyChecking=no" in cmd


class TestResolveSshKey:
    def test_returns_key_path_when_ssh_key_path_set_and_exists(self, tmp_path, monkeypatch):
        key_file = tmp_path / "id_rsa"
        key_file.write_text("FAKE KEY")
        monkeypatch.setenv("SSH_KEY_PATH", str(key_file))
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        result = _resolve_ssh_key()
        assert result == str(key_file)

    def test_returns_none_when_ssh_key_path_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SSH_KEY_PATH", str(tmp_path / "nonexistent.pem"))
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        result = _resolve_ssh_key()
        assert result is None

    def test_writes_temp_file_from_ssh_private_key(self, monkeypatch):
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.setenv("SSH_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----\n")
        result = _resolve_ssh_key()
        assert result is not None
        assert Path(result).exists()
        assert Path(result).read_text().startswith("-----BEGIN")
        # Permissions should be 0o600
        assert oct(Path(result).stat().st_mode)[-3:] == "600"

    def test_returns_none_when_no_ssh_vars(self, monkeypatch):
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        result = _resolve_ssh_key()
        assert result is None


# ---------------------------------------------------------------------------
# Clone helpers
# ---------------------------------------------------------------------------

class TestEnsureCloned:
    def test_returns_existing_path_when_already_cloned(self, tmp_path):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        with patch("services.indexer.workspace_resolver._CLONE_BASE", tmp_path):
            result = _ensure_cloned("my-repo", "git@github.com:org/repo.git", None)
        assert result == str(repo_dir)

    def test_clones_when_dir_missing(self, tmp_path):
        with patch("services.indexer.workspace_resolver._CLONE_BASE", tmp_path):
            with patch("services.indexer.workspace_resolver._clone_repo", return_value=True) as mock_clone:
                result = _ensure_cloned("new-repo", "git@github.com:org/new.git", "/key")
        mock_clone.assert_called_once_with(
            "git@github.com:org/new.git",
            tmp_path / "new-repo",
            "/key",
        )
        assert result == str(tmp_path / "new-repo")

    def test_returns_none_when_clone_fails(self, tmp_path):
        with patch("services.indexer.workspace_resolver._CLONE_BASE", tmp_path):
            with patch("services.indexer.workspace_resolver._clone_repo", return_value=False):
                result = _ensure_cloned("fail-repo", "git@github.com:org/fail.git", None)
        assert result is None


# ---------------------------------------------------------------------------
# load_repo_paths — local-mount path
# ---------------------------------------------------------------------------

class TestLoadRepoPathsLocalMount:
    def test_literal_paths_returned_when_they_exist(self, tmp_path):
        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()
        repo_b = tmp_path / "repo-b"
        repo_b.mkdir()
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "repo-a", "local_path": str(repo_a)},
                {"id": "repo-b", "local_path": str(repo_b)},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == [str(repo_a), str(repo_b)]

    def test_env_reference_resolved_when_path_exists(self, tmp_path, monkeypatch):
        repo_dir = tmp_path / "my-repo"
        repo_dir.mkdir()
        monkeypatch.setenv("MY_REPO_PATH", str(repo_dir))
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "my-repo", "local_path": "env:MY_REPO_PATH"}],
        )
        result = load_repo_paths(yaml_path)
        assert result == [str(repo_dir)]

    def test_missing_workspace_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_repo_paths(str(tmp_path / "nonexistent.yaml"))

    def test_empty_repos_list_raises(self, tmp_path):
        yaml_path = _write_workspace_yaml(tmp_path, [])
        with pytest.raises(ValueError, match="no repos\\[\\] entries"):
            load_repo_paths(yaml_path)


# ---------------------------------------------------------------------------
# load_repo_paths — ssh_url clone fallback
# ---------------------------------------------------------------------------

class TestLoadRepoPathsCloneFallback:
    def test_clones_when_local_path_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "remote-repo", "local_path": "/does/not/exist", "github": "git@github.com:org/remote.git"}],
        )
        cloned_path = tmp_path / "cloned"
        cloned_path.mkdir()
        with patch("services.indexer.workspace_resolver._ensure_cloned", return_value=str(cloned_path)) as mock:
            result = load_repo_paths(yaml_path)
        mock.assert_called_once_with("remote-repo", "git@github.com:org/remote.git", None)
        assert result == [str(cloned_path)]

    def test_uses_env_ssh_url_field(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "r", "ssh_url": "git@github.com:org/r.git"}],
        )
        with patch("services.indexer.workspace_resolver._ensure_cloned", return_value="/tmp/r") as mock:
            result = load_repo_paths(yaml_path)
        mock.assert_called_once_with("r", "git@github.com:org/r.git", None)
        assert result == ["/tmp/r"]

    def test_skipped_when_no_local_path_and_no_ssh_url(self, tmp_path):
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "no-url"},
                {"id": "ok", "local_path": str(tmp_path)},
            ],
        )
        result = load_repo_paths(yaml_path)
        assert result == [str(tmp_path)]

    def test_skipped_when_clone_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "fail", "github": "git@github.com:org/fail.git"}],
        )
        with patch("services.indexer.workspace_resolver._ensure_cloned", return_value=None):
            result = load_repo_paths(yaml_path)
        assert result == []

    def test_unset_env_local_path_falls_back_to_clone(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "r", "local_path": "env:MISSING_VAR", "github": "git@github.com:org/r.git"}],
        )
        with patch("services.indexer.workspace_resolver._ensure_cloned", return_value="/tmp/r") as mock:
            result = load_repo_paths(yaml_path)
        mock.assert_called_once_with("r", "git@github.com:org/r.git", None)
        assert result == ["/tmp/r"]

    def test_mixed_local_and_clone(self, tmp_path, monkeypatch):
        """One repo is mounted locally; one must be cloned."""
        local_repo = tmp_path / "local"
        local_repo.mkdir()
        monkeypatch.delenv("SSH_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [
                {"id": "local-r", "local_path": str(local_repo)},
                {"id": "remote-r", "local_path": "/not/mounted", "github": "git@github.com:org/remote.git"},
            ],
        )
        with patch("services.indexer.workspace_resolver._ensure_cloned", return_value="/tmp/remote-r"):
            result = load_repo_paths(yaml_path)
        assert result == [str(local_repo), "/tmp/remote-r"]

    def test_ssh_key_passed_to_ensure_cloned(self, tmp_path, monkeypatch):
        key_file = tmp_path / "id_rsa"
        key_file.write_text("FAKE KEY")
        monkeypatch.setenv("SSH_KEY_PATH", str(key_file))
        monkeypatch.delenv("SSH_PRIVATE_KEY", raising=False)
        yaml_path = _write_workspace_yaml(
            tmp_path,
            [{"id": "r", "github": "git@github.com:org/r.git"}],
        )
        with patch("services.indexer.workspace_resolver._ensure_cloned", return_value="/tmp/r") as mock:
            load_repo_paths(yaml_path)
        _, _, ssh_key_arg = mock.call_args[0]
        assert ssh_key_arg == str(key_file)
