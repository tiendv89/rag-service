"""Unit tests for services/indexer/git_watcher.py."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from services.indexer.git_watcher import GitWatcher


def _make_result(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


class TestGitWatcher:
    def test_initial_last_commit_is_none(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        assert watcher.last_commit is None

    # ------------------------------------------------------------------
    # changed_files — first call (no last_commit)
    # ------------------------------------------------------------------

    def test_first_call_returns_all_tracked_files(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stdout="README.md\nCLAUDE.md\n")
            result = watcher.changed_files()

        assert result == ["README.md", "CLAUDE.md"]
        # Should call git ls-files since no prior commit
        mock_run.assert_called_once_with(["git", "ls-files"])

    def test_first_call_empty_repo_returns_empty(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stdout="")
            result = watcher.changed_files()
        assert result == []

    # ------------------------------------------------------------------
    # changed_files — subsequent calls (last_commit set)
    # ------------------------------------------------------------------

    def test_subsequent_call_uses_diff(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        watcher._last_commit = "abc123"
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stdout="docs/features/f/product-spec.md\n")
            result = watcher.changed_files()

        mock_run.assert_called_once_with(["git", "diff", "--name-only", "abc123", "HEAD"])
        assert result == ["docs/features/f/product-spec.md"]

    def test_diff_failure_falls_back_to_ls_files(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        watcher._last_commit = "abc123"

        ls_result = _make_result(stdout="README.md\n")
        diff_result = _make_result(stderr="some error", returncode=1)

        with patch.object(watcher, "_run") as mock_run:
            def side_effect(cmd):
                if "diff" in cmd:
                    return diff_result
                return ls_result
            mock_run.side_effect = side_effect
            result = watcher.changed_files()

        assert result == ["README.md"]

    # ------------------------------------------------------------------
    # advance
    # ------------------------------------------------------------------

    def test_advance_stores_head_sha(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stdout="deadbeef1234\n")
            watcher.advance()
        assert watcher.last_commit == "deadbeef1234"

    def test_advance_failure_does_not_raise(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stderr="error", returncode=128)
            watcher.advance()  # should not raise
        assert watcher.last_commit is None

    # ------------------------------------------------------------------
    # pull
    # ------------------------------------------------------------------

    def test_pull_fetches_and_hard_resets_to_upstream(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))

        def side_effect(cmd, use_ssh=False):
            if "rev-parse" in cmd:
                return _make_result(stdout="origin/main\n")
            return _make_result()

        with patch.object(watcher, "_run") as mock_run:
            mock_run.side_effect = side_effect
            watcher.pull()

        mock_run.assert_has_calls(
            [
                call(["git", "fetch", "origin"], use_ssh=True),
                call(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]),
                call(["git", "reset", "--hard", "origin/main"]),
            ]
        )

    def test_pull_resets_to_origin_head_when_no_upstream(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))

        def side_effect(cmd, use_ssh=False):
            if "rev-parse" in cmd:
                # No upstream configured — rev-parse fails.
                return _make_result(stderr="no upstream", returncode=128)
            return _make_result()

        with patch.object(watcher, "_run") as mock_run:
            mock_run.side_effect = side_effect
            watcher.pull()

        mock_run.assert_has_calls(
            [call(["git", "reset", "--hard", "origin/HEAD"])]
        )

    def test_pull_skips_reset_when_fetch_fails(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))

        def side_effect(cmd, use_ssh=False):
            if "fetch" in cmd:
                return _make_result(stderr="network error", returncode=1)
            raise AssertionError("reset must not run when fetch fails")

        with patch.object(watcher, "_run") as mock_run:
            mock_run.side_effect = side_effect
            watcher.pull()  # should not raise

        mock_run.assert_called_once_with(["git", "fetch", "origin"], use_ssh=True)

    def test_pull_failure_does_not_raise(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stderr="not a git repo", returncode=128)
            watcher.pull()  # should not raise

    # ------------------------------------------------------------------
    # _all_tracked_files
    # ------------------------------------------------------------------

    def test_all_tracked_files_filters_blank_lines(self, tmp_path):
        watcher = GitWatcher(str(tmp_path))
        with patch.object(watcher, "_run") as mock_run:
            mock_run.return_value = _make_result(stdout="file1.md\n\n\nfile2.md\n")
            result = watcher._all_tracked_files()
        assert result == ["file1.md", "file2.md"]
