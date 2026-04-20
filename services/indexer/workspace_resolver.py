"""
Resolve repo paths from workspace.yaml for the indexer.

Reads workspace.yaml, iterates repos[], and resolves container-internal
paths for each repo. Repos whose local_path env reference is unset are
skipped with a warning so the indexer degrades gracefully.
"""

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_repo_paths(workspace_yaml_path: str) -> list[str]:
    """
    Parse workspace.yaml and return resolved container-internal paths for all repos.

    local_path values follow one of two formats:
    - ``env:VAR_NAME`` — resolved from the named environment variable
    - A literal path string — used as-is

    Repos with an unresolvable env reference are skipped (warning logged).
    Repos with no local_path are skipped (warning logged).

    Args:
        workspace_yaml_path: Filesystem path to workspace.yaml inside the container.

    Returns:
        List of resolved, non-empty repo paths.

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

    resolved: list[str] = []
    for repo in repos:
        repo_id = repo.get("id", "<unknown>")
        local_path: str = repo.get("local_path", "")

        if not local_path:
            logger.warning("Repo %r has no local_path — skipping", repo_id)
            continue

        if local_path.startswith("env:"):
            var_name = local_path[4:]
            value = os.environ.get(var_name)
            if not value:
                logger.warning(
                    "Repo %r: env var %s is unset or empty — skipping",
                    repo_id,
                    var_name,
                )
                continue
            resolved.append(value)
            logger.debug("Repo %r → %s (from env %s)", repo_id, value, var_name)
        else:
            resolved.append(local_path)
            logger.debug("Repo %r → %s (literal)", repo_id, local_path)

    return resolved
