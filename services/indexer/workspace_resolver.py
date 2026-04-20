"""
Resolve repo paths from workspace.yaml for the indexer.

Reads workspace.yaml, iterates repos[], and returns a usable local path for
each repo using the following clone-or-pull algorithm:

  1. Resolve local_path (literal string or ``env:VAR_NAME`` reference).
  2. If the resolved path exists on the filesystem → use it directly.
  3. If local_path is absent, empty, or does not exist on the filesystem →
     clone from repos[].ssh_url into /tmp/indexer-repos/<repo_id>/.

SSH authentication for step 3 uses (in order of preference):
  - SSH_KEY_PATH env var: path to an SSH private key file
  - SSH_PRIVATE_KEY env var: raw PEM content written to a temp file

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


def _resolve_ssh_key() -> str | None:
    """
    Return the path to a usable SSH private key, or None if unavailable.

    Checks SSH_KEY_PATH first; if absent checks SSH_PRIVATE_KEY (raw content)
    and writes it to a temp file.  The temp file is not cleaned up because the
    indexer is a long-running process and the file must survive across calls.
    """
    key_path = os.environ.get("SSH_KEY_PATH", "")
    if key_path and Path(key_path).exists():
        return key_path

    raw_key = os.environ.get("SSH_PRIVATE_KEY", "")
    if raw_key:
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

def load_repo_paths(workspace_yaml_path: str) -> list[str]:
    """
    Parse workspace.yaml and return resolved local paths for all repos.

    Resolution algorithm (per repo entry):
    1. Resolve local_path (literal or ``env:VAR_NAME`` reference).
    2. If the resolved path exists on the filesystem → use it directly.
    3. If local_path is absent/unresolvable/non-existent → clone from
       repos[].ssh_url into /tmp/indexer-repos/<repo_id>/ using SSH key
       from SSH_KEY_PATH or SSH_PRIVATE_KEY env vars.
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

    ssh_key_path = _resolve_ssh_key()
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
