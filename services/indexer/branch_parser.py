"""Parse Git branch names into (feature_id, task_id) tuples."""

import re

_PATTERN = re.compile(r"feature/(?P<feature_id>[^/]+?)(?:-T(?P<task_id>\d+))?$")


def parse_branch(branch_name: str) -> tuple[str | None, str | None]:
    """
    Parse a branch name into (feature_id, task_id).

    Returns (None, None) if the branch does not match the
    ``feature/<feature_id>[-T<n>]`` pattern.

    Examples:
        "feature/agent-rag-pr-index-T1" → ("agent-rag-pr-index", "1")
        "feature/agent-rag-pr-index"    → ("agent-rag-pr-index", None)
        "main"                          → (None, None)
    """
    if not branch_name:
        return None, None
    m = _PATTERN.search(branch_name)
    if not m:
        return None, None
    return m.group("feature_id"), m.group("task_id")
