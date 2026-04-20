"""
Indexer service entry point.

Environment variables:
  QDRANT_URL                     URL of the Qdrant instance (required)
  WORKSPACE_ID                   Workspace partition key (required)
  WORKSPACE_YAML_PATH            Path to workspace.yaml mounted inside the container
                                 (required; e.g. /workspace/workspace.yaml)
  INDEXER_POLL_INTERVAL_SECONDS  Polling interval in seconds (default: 300)
  SSH_KEY_PATH                   Path to SSH private key for cloning repos (optional;
                                 used when repos are not pre-mounted in k8s/cloud)
  SSH_PRIVATE_KEY                Raw PEM content of an SSH private key (alternative
                                 to SSH_KEY_PATH; written to a temp file at startup)

  Repo paths are resolved from workspace.yaml → repos[] at startup using a
  clone-or-pull strategy:
    1. If repos[].local_path exists on the filesystem → use it directly.
    2. Otherwise → clone from repos[].github (ssh_url) into
       /tmp/indexer-repos/<repo_id>/ using SSH_KEY_PATH or SSH_PRIVATE_KEY.

The indexer:
  1. Reads workspace.yaml to discover which repos to watch.
  2. Initialises the Qdrant collection for the workspace (idempotent).
  3. On each cycle:
     a. git pull in every watched repo.
     b. Detect changed files via git diff since last cycle.
     c. Classify each file by source_type.
     d. Chunk and embed changed files.
     e. Upsert points to Qdrant.
     f. Advance the git watermark.
  4. Sleep for INDEXER_POLL_INTERVAL_SECONDS, then repeat.
"""

import hashlib
import logging
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient

from services.indexer.chunker import chunk_document
from services.indexer.embedder import Embedder
from services.indexer.git_watcher import GitWatcher
from services.indexer.source_mapper import classify_path
from services.indexer.workspace_resolver import load_repo_paths
from services.shared.qdrant_init import init_collection, upsert_points
from services.shared.schema import ChunkPayload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Sentinel for graceful shutdown
_SHUTDOWN = False


def _handle_sigterm(signum, frame) -> None:  # noqa: ANN001
    global _SHUTDOWN
    _SHUTDOWN = True
    logger.info("Received SIGTERM — shutting down after current cycle.")


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Point ID generation
# ---------------------------------------------------------------------------

def _point_id(repo_path: str, rel_path: str, chunk_index: int) -> str:
    """
    Deterministic point ID derived from repo path, file path, and chunk index.

    Using a hash ensures the same file+chunk always maps to the same ID,
    which allows Qdrant upsert to be truly idempotent (overwrite on re-index).
    """
    key = f"{repo_path}|{rel_path}|{chunk_index}"
    hex_digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=hex_digest))


# ---------------------------------------------------------------------------
# Indexing a single repo
# ---------------------------------------------------------------------------

def index_repo(
    *,
    repo_path: str,
    watcher: GitWatcher,
    embedder: Embedder,
    client: QdrantClient,
    workspace_id: str,
) -> int:
    """
    Index changed files from one repo. Returns the number of points upserted.
    """
    changed = watcher.changed_files()
    if not changed:
        logger.debug("No changed files in %s", repo_path)
        return 0

    logger.info("%d changed file(s) to consider in %s", len(changed), repo_path)

    indexed_at = datetime.now(timezone.utc).isoformat()
    points: list[dict] = []

    for rel_path in changed:
        abs_path = Path(repo_path) / rel_path
        classification = classify_path(rel_path)
        if classification is None:
            logger.debug("Skipping unindexed path: %s", rel_path)
            continue

        source_type, feature_id = classification

        if not abs_path.exists():
            logger.debug("Path no longer exists (deleted?): %s", abs_path)
            continue

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", abs_path, exc)
            continue

        chunks = chunk_document(source_type, content)
        if not chunks:
            logger.debug("No chunks produced for %s", rel_path)
            continue

        for chunk_index, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            payload = ChunkPayload(
                workspace_id=workspace_id,
                source_type=source_type,
                source_path=rel_path,
                chunk_index=chunk_index,
                indexed_at=indexed_at,
                feature_id=feature_id,
            )

            points.append(
                {
                    "id": _point_id(repo_path, rel_path, chunk_index),
                    "text": chunk_text,
                    "payload": payload.to_dict(),
                }
            )

    if not points:
        return 0

    # Embed all chunks in one batch for efficiency
    vectors = embedder.encode([p["text"] for p in points])

    qdrant_points = [
        {
            "id": p["id"],
            "vector": v,
            "payload": p["payload"],
        }
        for p, v in zip(points, vectors)
    ]

    upsert_points(client, workspace_id, qdrant_points)
    logger.info("Upserted %d point(s) from %s", len(qdrant_points), repo_path)
    return len(qdrant_points)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    qdrant_url: str,
    workspace_id: str,
    repo_paths: list[str],
    poll_interval: int = 300,
    embedder: Optional[Embedder] = None,
) -> None:
    """
    Run the indexer polling loop.

    This function blocks until a SIGTERM/SIGINT is received.
    """
    client = QdrantClient(url=qdrant_url)

    # Ensure the workspace collection exists (idempotent)
    init_collection(client, workspace_id)
    logger.info(
        "Indexer started — workspace=%r, repos=%r, poll=%ds",
        workspace_id,
        repo_paths,
        poll_interval,
    )

    if embedder is None:
        embedder = Embedder()

    watchers = {rp: GitWatcher(rp) for rp in repo_paths}

    while not _SHUTDOWN:
        cycle_start = time.monotonic()

        for repo_path, watcher in watchers.items():
            try:
                watcher.pull()
                index_repo(
                    repo_path=repo_path,
                    watcher=watcher,
                    embedder=embedder,
                    client=client,
                    workspace_id=workspace_id,
                )
                watcher.advance()
            except Exception:
                logger.exception("Error indexing repo %s — skipping this cycle", repo_path)

        elapsed = time.monotonic() - cycle_start
        sleep_time = max(0.0, poll_interval - elapsed)
        logger.info(
            "Cycle complete in %.1fs — sleeping %.1fs before next cycle",
            elapsed,
            sleep_time,
        )

        # Sleep in small increments so SIGTERM is handled promptly
        deadline = time.monotonic() + sleep_time
        while not _SHUTDOWN and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    logger.info("Indexer stopped.")


def main() -> None:
    qdrant_url = os.environ["QDRANT_URL"]
    workspace_id = os.environ["WORKSPACE_ID"]

    workspace_yaml_path = os.environ.get("WORKSPACE_YAML_PATH", "")
    if not workspace_yaml_path:
        raise ValueError(
            "WORKSPACE_YAML_PATH must be set — path to the mounted workspace.yaml"
        )

    repo_paths = load_repo_paths(workspace_yaml_path)
    if not repo_paths:
        raise ValueError(
            f"No repo paths resolved from {workspace_yaml_path} — "
            "check that repos[].local_path env vars are set in the container"
        )

    poll_interval = int(os.environ.get("INDEXER_POLL_INTERVAL_SECONDS", "300"))

    run(
        qdrant_url=qdrant_url,
        workspace_id=workspace_id,
        repo_paths=repo_paths,
        poll_interval=poll_interval,
    )


if __name__ == "__main__":
    main()
