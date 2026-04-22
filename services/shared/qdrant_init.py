"""
Idempotent Qdrant collection initialisation.

Creates a collection keyed by workspace_id if it does not already exist.
All collections use 768-dimensional vectors (BAAI/bge-base-en-v1.5).
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from services.shared.schema import build_workspace_filter

logger = logging.getLogger(__name__)

# Vector dimension for BAAI/bge-base-en-v1.5
VECTOR_DIM = 768

# Default distance metric
DISTANCE = qdrant_models.Distance.COSINE


def collection_name_for(workspace_id: str) -> str:
    """
    Return the Qdrant collection name for a given workspace_id.

    Raises ValueError if workspace_id is empty.
    """
    if not workspace_id:
        raise ValueError("workspace_id is required")
    return workspace_id


def init_collection(client: QdrantClient, workspace_id: str) -> bool:
    """
    Create the Qdrant collection for workspace_id if it does not exist.

    This function is idempotent — calling it multiple times with the same
    workspace_id is safe. Returns True if the collection was created, False
    if it already existed.

    Raises ValueError if workspace_id is empty.
    """
    if not workspace_id:
        raise ValueError("workspace_id is required")

    collection = collection_name_for(workspace_id)

    try:
        info = client.get_collection(collection_name=collection)
        existing_size = info.config.params.vectors.size
        if existing_size != VECTOR_DIM:
            raise RuntimeError(
                f"Collection {collection!r} has vector_size={existing_size} but "
                f"VECTOR_DIM={VECTOR_DIM}. Drop and recreate the collection before "
                "deploying: stop indexer → DELETE /collections/{workspace_id} → "
                "redeploy → re-index."
            )
        logger.debug("Collection %r already exists — skipping creation", collection)
        return False
    except RuntimeError:
        raise
    except Exception as exc:
        # qdrant-client raises UnexpectedResponse (404) when the collection
        # does not exist. Re-raise anything that is not a 404.
        if not _is_not_found(exc):
            raise

    client.create_collection(
        collection_name=collection,
        vectors_config=qdrant_models.VectorParams(
            size=VECTOR_DIM,
            distance=DISTANCE,
        ),
    )

    # Create a payload index on workspace_id for efficient filtering.
    client.create_payload_index(
        collection_name=collection,
        field_name="workspace_id",
        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
    )

    logger.info("Created Qdrant collection %r for workspace %r", collection, workspace_id)
    return True


def upsert_points(
    client: QdrantClient,
    workspace_id: str,
    points: list[dict],
) -> None:
    """
    Upsert a list of points into the workspace collection.

    Each point dict must contain:
        id       — unique point ID (str or int)
        vector   — list[float] of length VECTOR_DIM
        payload  — dict produced by ChunkPayload.to_dict()

    Raises ValueError if workspace_id is missing or empty, or if any point's
    payload is missing workspace_id.
    """
    if not workspace_id:
        raise ValueError("workspace_id is required for all upserts")

    for i, point in enumerate(points):
        payload = point.get("payload", {})
        if not payload.get("workspace_id"):
            raise ValueError(
                f"Point at index {i} is missing workspace_id in its payload; "
                "every upserted point must carry workspace_id"
            )

    collection = collection_name_for(workspace_id)

    qdrant_points = [
        qdrant_models.PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p["payload"],
        )
        for p in points
    ]

    client.upsert(
        collection_name=collection,
        points=qdrant_points,
    )
    logger.debug("Upserted %d point(s) to collection %r", len(qdrant_points), collection)


def query_points(
    client: QdrantClient,
    workspace_id: str,
    query_vector: list[float],
    top_k: int = 5,
    source_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Query the workspace collection for the top-k most similar vectors.

    workspace_id is required and is applied as a must-filter on every search.
    Raises ValueError if workspace_id is missing or empty.

    Optional source_types further restricts results to the given source type
    values.

    Returns a list of dicts with keys: id, score, payload.
    """
    if not workspace_id:
        raise ValueError(
            "workspace_id is required for all queries; "
            "queries without workspace_id are not permitted"
        )

    workspace_filter = build_workspace_filter(workspace_id)

    if source_types:
        workspace_filter["must"].append(
            {
                "key": "source_type",
                "match": {"any": source_types},
            }
        )

    collection = collection_name_for(workspace_id)

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_models.Filter(**workspace_filter),
        with_payload=True,
    )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results.points
    ]


def _is_not_found(exc: Exception) -> bool:
    """Return True if exc represents a 404 / collection-not-found error."""
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code == 404
    # Fallback: check message text for older client versions
    msg = str(exc).lower()
    return "not found" in msg or "doesn't exist" in msg or "404" in msg
