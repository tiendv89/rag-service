"""
Git-based changed-file detection for the indexer.

Uses `git diff --name-only` to find files that changed since the last
indexed commit. On the first run (no prior commit recorded) every tracked
file is treated as changed so a full initial index is performed.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GitWatcher:
    """
    Track which files changed in a git repo since the last indexed HEAD.

    Usage::

        watcher = GitWatcher(repo_path="/path/to/repo", ssh_key_path="/home/agent/.ssh/id_rsa")
        changed = watcher.changed_files()  # returns list[str] of rel paths
        watcher.advance()                   # record current HEAD as last_seen
    """

    def __init__(self, repo_path: str, ssh_key_path: Optional[str] = None) -> None:
        self._repo = Path(repo_path).resolve()
        self._last_commit: Optional[str] = None
        self._ssh_key_path = ssh_key_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pull(self) -> None:
        """
        Run `git pull` in the repo to fetch the latest changes.

        Failures are logged as warnings but do not raise — the indexer
        continues with the locally available state.
        """
        result = self._run(["git", "pull", "--ff-only"], use_ssh=True)
        if result.returncode != 0:
            logger.warning(
                "git pull failed in %s (exit %d): %s",
                self._repo,
                result.returncode,
                result.stderr.strip(),
            )

    def changed_files(self) -> list[str]:
        """
        Return relative paths of files that changed since last advance().

        On the first call (no last_commit) all tracked files are returned
        so the indexer performs a full initial index.
        """
        if self._last_commit is None:
            return self._all_tracked_files()

        result = self._run(
            ["git", "diff", "--name-only", self._last_commit, "HEAD"]
        )
        if result.returncode != 0:
            logger.warning(
                "git diff failed in %s: %s; falling back to full index",
                self._repo,
                result.stderr.strip(),
            )
            return self._all_tracked_files()

        return [line for line in result.stdout.splitlines() if line.strip()]

    def advance(self) -> None:
        """Record the current HEAD commit as the new last_seen baseline."""
        result = self._run(["git", "rev-parse", "HEAD"])
        if result.returncode == 0:
            self._last_commit = result.stdout.strip()
            logger.debug("Advanced last_commit to %s in %s", self._last_commit, self._repo)
        else:
            logger.warning(
                "git rev-parse HEAD failed in %s: %s",
                self._repo,
                result.stderr.strip(),
            )

    @property
    def last_commit(self) -> Optional[str]:
        """The last commit SHA that was fully indexed, or None."""
        return self._last_commit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_tracked_files(self) -> list[str]:
        """Return all files tracked by git in the repo."""
        result = self._run(["git", "ls-files"])
        if result.returncode != 0:
            logger.warning(
                "git ls-files failed in %s: %s",
                self._repo,
                result.stderr.strip(),
            )
            return []
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _run(self, cmd: list[str], use_ssh: bool = False) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if use_ssh and self._ssh_key_path:
            env["GIT_SSH_COMMAND"] = (
                f"ssh -i {self._ssh_key_path} "
                "-o StrictHostKeyChecking=no "
                "-o UserKnownHostsFile=/dev/null"
            )
        return subprocess.run(
            cmd,
            cwd=self._repo,
            env=env,
            capture_output=True,
            text=True,
        )
