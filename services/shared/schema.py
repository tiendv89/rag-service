"""
Canonical payload schema for all documents indexed in Qdrant.

Every point stored in Qdrant carries this payload. workspace_id is required
on all upserts and must be present as a filter on all queries.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# Valid source types
VALID_SOURCE_TYPES = frozenset({
    "skill",
    "task_log",
    "product_spec",
    "technical_design",
    "readme",
    "claude_md",
})


@dataclass
class ChunkPayload:
    """
    Payload attached to every Qdrant point.

    workspace_id is the tenant partition key and is required — any code
    path that constructs a ChunkPayload without it must raise ValueError.
    """

    workspace_id: str
    source_type: str
    source_path: str
    chunk_index: int
    indexed_at: str
    feature_id: Optional[str] = None
    content: str = ""

    def __post_init__(self) -> None:
        if not self.workspace_id:
            raise ValueError("workspace_id is required and must not be empty")
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, "
                f"got: {self.source_type!r}"
            )
        if not self.source_path:
            raise ValueError("source_path is required and must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")

    def to_dict(self) -> dict:
        """Return payload as a plain dict suitable for Qdrant point payload."""
        d = {
            "workspace_id": self.workspace_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "feature_id": self.feature_id,
            "chunk_index": self.chunk_index,
            "indexed_at": self.indexed_at,
        }
        if self.content:
            d["content"] = self.content
        return d

    @classmethod
    def now_iso(cls) -> str:
        """Return current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()


def build_workspace_filter(workspace_id: str) -> dict:
    """
    Build a Qdrant must-filter that restricts results to a single workspace.

    Raises ValueError if workspace_id is empty — queries without a
    workspace_id filter are not permitted.

    Returns a dict matching the Qdrant Filter structure:
        {"must": [{"key": "workspace_id", "match": {"value": "<id>"}}]}
    """
    if not workspace_id:
        raise ValueError(
            "workspace_id is required for all queries; "
            "queries without a workspace_id filter are not permitted"
        )
    return {
        "must": [
            {
                "key": "workspace_id",
                "match": {"value": workspace_id},
            }
        ]
    }
