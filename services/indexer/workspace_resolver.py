"""
Resolve repo paths from workspace.yaml for the indexer.

Reads workspace.yaml, iterates repos[], and returns a usable local path for
each repo using the following clone-or-pull algorithm:

  1. Resolve local_path (literal string or ``env:VAR_NAME`` reference).
  2. If the resolved path exists on the filesystem → use it directly.
  3. If local_path is absent, empty, or does not exist on the filesystem →
     clone from repos[].ssh_url into /tmp/indexer-repos/<repo_id>/.

SSH authentication for step 3 uses the SSH_PRIVATE_KEY env var (raw PEM
content written to a temp file at startup).

Repos with no local_path AND no ssh_url are skipped with a warning.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CLONE_BASE = Path("/tmp/indexer-repos")
_WORKSPACE_CLONE_BASE = Path("/tmp/indexer-workspace")


# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------

def _build_git_ssh_command(ssh_key_path: str) -> str:
    """Return a GIT_SSH_COMMAND value that uses the given key file."""
    return (
        f"ssh -i {ssh_key_path} "
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null"
    )


def resolve_ssh_key() -> str | None:
    """
    Return the path to a usable SSH private key, or None if unavailable.

    Reads SSH_PRIVATE_KEY (raw PEM content) and writes it to a temp file.
    The temp file is not cleaned up because the indexer is a long-running
    process and the file must survive across calls.
    """
    raw_key = os.environ.get("SSH_PRIVATE_KEY", "")
    if raw_key:
        # Normalize literal \n (from Docker Compose env_file / YAML interpolation)
        raw_key = raw_key.replace("\\n", "\n")
        tmp = tempfile.NamedTemporaryFile(
            prefix="indexer_ssh_key_",
            mode="w",
            delete=False,
            suffix=".pem",
        )
        tmp.write(raw_key)
        tmp.flush()
        tmp.close()
        os.chmod(tmp.name, 0o600)
        logger.debug("SSH_PRIVATE_KEY written to temp file %s", tmp.name)
        return tmp.name

    return None


# ---------------------------------------------------------------------------
# Clone / pull helpers
# ---------------------------------------------------------------------------

def _clone_repo(ssh_url: str, dest: Path, ssh_key_path: str | None) -> bool:
    """
    Clone ssh_url into dest.  Returns True on success, False on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if ssh_key_path:
        env["GIT_SSH_COMMAND"] = _build_git_ssh_command(ssh_key_path)

    logger.info("Cloning %s → %s", ssh_url, dest)
    result = subprocess.run(
        ["git", "clone", ssh_url, str(dest)],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            "git clone failed for %s (exit %d): %s",
            ssh_url,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


def _pull_repo(dest: Path, ssh_key_path: str | None) -> bool:
    """
    Refresh an already-cloned repo via fetch + hard-reset so that rebases and
    force-pushes on origin are always reflected. Returns True on success.

    Failures are logged as warnings (not raised) so a transient network error
    falls back to the already-checked-out content rather than crashing.
    """
    env = os.environ.copy()
    if ssh_key_path:
        env["GIT_SSH_COMMAND"] = _build_git_ssh_command(ssh_key_path)

    for args in (["fetch", "origin"], ["reset", "--hard", "FETCH_HEAD"]):
        result = subprocess.run(
            ["git", "-C", str(dest), *args],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "git %s failed for %s: %s",
                args[0],
                dest,
                result.stderr.strip(),
            )
            return False
    return True


def bootstrap_workspace(
    workspace_url: str,
    base_dir: str | Path = _WORKSPACE_CLONE_BASE,
    ssh_key_path: str | None = None,
) -> str:
    """
    Clone (or refresh) the workspace management repo and return the path to
    its ``workspace.yaml``.

    This mirrors the GitNexus indexer: the RAG indexer no longer needs
    ``workspace.yaml`` bind-mounted from the host. It clones the management
    repo from ``workspace_url`` and reads ``workspace.yaml`` from inside the
    clone, so a fresh copy is picked up on every restart.

    Args:
        workspace_url: SSH (or HTTPS) URL of the workspace management repo.
        base_dir: Directory the workspace repo is cloned into.
        ssh_key_path: Path to an SSH private key file (from SSH_PRIVATE_KEY).

    Returns:
        Absolute path to ``workspace.yaml`` inside the cloned repo.

    Raises:
        ValueError: If ``workspace_url`` is empty.
        RuntimeError: If the repo cannot be cloned.
        FileNotFoundError: If the clone contains no ``workspace.yaml``.
    """
    if not workspace_url:
        raise ValueError("WORKSPACE_URL is required to clone the workspace repo")

    dest = Path(base_dir) / "_workspace"

    if dest.exists() and (dest / ".git").exists():
        logger.debug("Workspace repo already cloned at %s — refreshing", dest)
        _pull_repo(dest, ssh_key_path)
    elif not _clone_repo(workspace_url, dest, ssh_key_path):
        raise RuntimeError(f"Failed to clone workspace repo from {workspace_url}")

    yaml_path = dest / "workspace.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"workspace.yaml not found in workspace repo cloned from "
            f"{workspace_url} (expected at {yaml_path})"
        )
    logger.info("Resolved workspace.yaml from %s → %s", workspace_url, yaml_path)
    return str(yaml_path)


def _ensure_cloned(repo_id: str, ssh_url: str, ssh_key_path: str | None) -> str | None:
    """
    Ensure repo is cloned into /tmp/indexer-repos/<repo_id>/.

    If the directory already exists it is reused (git pull happens each cycle
    via GitWatcher).  Returns the local path on success, None on failure.
    """
    dest = _CLONE_BASE / repo_id
    if dest.exists() and (dest / ".git").exists():
        logger.debug("Repo %r already cloned at %s", repo_id, dest)
        return str(dest)

    if not _clone_repo(ssh_url, dest, ssh_key_path):
        return None
    return str(dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_repo_paths(workspace_yaml_path: str, ssh_key_path: str | None = None) -> list[str]:
    """
    Parse workspace.yaml and return resolved local paths for all repos.

    Resolution algorithm (per repo entry):
    1. Resolve local_path (literal or ``env:VAR_NAME`` reference).
    2. If the resolved path exists on the filesystem → use it directly.
    3. If local_path is absent/unresolvable/non-existent → clone from
       repos[].ssh_url into /tmp/indexer-repos/<repo_id>/ using SSH key
       from SSH_PRIVATE_KEY env var.
    4. Repos with neither a usable local_path nor an ssh_url are skipped.

    Args:
        workspace_yaml_path: Filesystem path to workspace.yaml.

    Returns:
        List of resolved local repo paths.

    Raises:
        FileNotFoundError: If workspace_yaml_path does not exist.
        ValueError: If workspace.yaml contains no repos[] entries.
    """
    path = Path(workspace_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"workspace.yaml not found at {workspace_yaml_path}")

    with path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    repos = config.get("repos", [])
    if not repos:
        raise ValueError(f"workspace.yaml at {workspace_yaml_path} has no repos[] entries")

    if ssh_key_path is None:
        ssh_key_path = resolve_ssh_key()
    resolved: list[str] = []

    for repo in repos:
        repo_id = repo.get("id", "<unknown>")
        local_path_raw: str = repo.get("local_path", "")
        ssh_url: str = repo.get("github", repo.get("ssh_url", ""))

        # Step 1: resolve local_path
        resolved_local: str = ""
        if local_path_raw:
            if local_path_raw.startswith("env:"):
                var_name = local_path_raw[4:]
                resolved_local = os.environ.get(var_name, "")
                if not resolved_local:
                    logger.debug(
                        "Repo %r: env var %s is unset — will try ssh_url fallback",
                        repo_id,
                        var_name,
                    )
            else:
                resolved_local = local_path_raw

        # Step 2: use local_path if the directory exists
        if resolved_local and Path(resolved_local).exists():
            resolved.append(resolved_local)
            logger.debug("Repo %r → %s (local mount)", repo_id, resolved_local)
            continue

        # Step 3: clone fallback
        if ssh_url:
            cloned_path = _ensure_cloned(repo_id, ssh_url, ssh_key_path)
            if cloned_path:
                resolved.append(cloned_path)
                logger.info(
                    "Repo %r → %s (cloned from %s)", repo_id, cloned_path, ssh_url
                )
                continue
            logger.warning(
                "Repo %r: clone from %s failed — skipping", repo_id, ssh_url
            )
        else:
            logger.warning(
                "Repo %r: local_path %r does not exist and no ssh_url/github — skipping",
                repo_id,
                resolved_local or local_path_raw,
            )

    return resolved
