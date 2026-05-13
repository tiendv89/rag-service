"""
Fetch and index merged GitHub PR descriptions as source_type='pr_description'.

Algorithm per repo:
1. Load cursor (last_indexed_merged_at) from pr_index_state.json.
2. Fetch closed PRs from GitHub API sorted ascending by update time;
   filter to merged_at is not null; skip any with merged_at <= cursor.
3. For each new PR, chunk the document, embed, and upsert to Qdrant.
4. Advance cursor atomically (write-to-tmp then rename).

Cold start (missing state file) triggers full re-index — safe because upsert
is idempotent. Absence of GITHUB_TOKEN must be handled by the caller; this
module assumes a valid token is provided.
"""

import hashlib
import json
import logging
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from services.indexer.branch_parser import parse_branch
from services.indexer.chunker import chunk_document
from services.shared.qdrant_init import upsert_points

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


def _point_id(repo_id: str, pr_number: int, chunk_index: int) -> str:
    """Deterministic UUID from repo + PR number + chunk index."""
    key = f"pr|{repo_id}|{pr_number}|{chunk_index}"
    hex_digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=hex_digest))


class PrIndexer:
    """Fetch and index merged PRs for one or more repos."""

    def __init__(
        self,
        github_token: str,
        qdrant_client,
        embedder,
        workspace_id: str,
        state_path: str = "pr_index_state.json",
    ) -> None:
        self._token = github_token
        self._client = qdrant_client
        self._embedder = embedder
        self._workspace_id = workspace_id
        self._state_path = Path(state_path)
        self._state: dict = self._load_state()

    # ------------------------------------------------------------------
    # Cursor persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Cannot read PR index state from %s: %s — starting fresh", self._state_path, exc)
        return {}

    def _save_state(self) -> None:
        """Write state atomically via tmp file + rename."""
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        tmp.rename(self._state_path)

    # ------------------------------------------------------------------
    # GitHub API
    # ------------------------------------------------------------------

    def _github_get(self, url: str) -> list | dict:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "rag-service-pr-indexer/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch_merged_prs(self, repo_full_name: str, cursor: Optional[str]) -> list[dict]:
        """
        Return merged PRs with merged_at > cursor, paginating as needed.

        Sorted ascending so cursor advances monotonically.
        """
        results: list[dict] = []
        page = 1

        while True:
            url = (
                f"{_GITHUB_API}/repos/{repo_full_name}/pulls"
                f"?state=closed&sort=updated&direction=asc&per_page=100&page={page}"
            )
            try:
                data = self._github_get(url)
            except urllib.error.HTTPError as exc:
                logger.warning("GitHub API HTTP %s for %s page %d: %s", exc.code, repo_full_name, page, exc)
                break
            except Exception as exc:
                logger.warning("GitHub API error for %s page %d: %s", repo_full_name, page, exc)
                break

            if not data:
                break

            for pr in data:
                merged_at = pr.get("merged_at")
                if not merged_at:
                    continue
                if cursor and merged_at <= cursor:
                    continue
                results.append(pr)

            if len(data) < 100:
                break
            page += 1

        return results

    # ------------------------------------------------------------------
    # Public indexing method
    # ------------------------------------------------------------------

    def index_repo_prs(self, repo_full_name: str, repo_id: str) -> int:
        """
        Fetch merged PRs not yet indexed; embed and upsert.

        Returns the count of newly indexed PRs (not chunks).
        """
        cursor: Optional[str] = self._state.get(repo_id)

        prs = self._fetch_merged_prs(repo_full_name, cursor)
        if not prs:
            logger.debug("No new merged PRs for %s", repo_full_name)
            return 0

        indexed_at = datetime.now(timezone.utc).isoformat()
        points: list[dict] = []
        latest_merged_at: Optional[str] = cursor

        for pr in prs:
            pr_number: int = pr["number"]
            title: str = pr.get("title") or ""
            body: str = pr.get("body") or ""
            merged_at: str = pr["merged_at"]
            branch_name: str = pr.get("head", {}).get("ref", "")

            feature_id, task_id = parse_branch(branch_name)

            document = f"# {title}\n\n{body}".strip()
            chunks = chunk_document("pr_description", document)

            for chunk_index, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                payload = {
                    "source_type": "pr_description",
                    "source_path": f"github.com/{repo_full_name}/pull/{pr_number}",
                    "workspace_id": self._workspace_id,
                    "repo_id": repo_id,
                    "pr_number": pr_number,
                    "feature_id": feature_id,
                    "task_id": task_id,
                    "pr_title": title,
                    "merged_at": merged_at,
                    "chunk_index": chunk_index,
                    "indexed_at": indexed_at,
                    "content": chunk_text,
                }
                points.append({
                    "id": _point_id(repo_id, pr_number, chunk_index),
                    "text": chunk_text,
                    "payload": payload,
                })

            if latest_merged_at is None or merged_at > latest_merged_at:
                latest_merged_at = merged_at

        if not points:
            return 0

        vectors = self._embedder.encode([p["text"] for p in points])
        qdrant_points = [
            {"id": p["id"], "vector": v, "payload": p["payload"]}
            for p, v in zip(points, vectors)
        ]

        upsert_points(self._client, self._workspace_id, qdrant_points)

        if latest_merged_at:
            self._state[repo_id] = latest_merged_at
            self._save_state()

        logger.info("Indexed %d new PR(s) for %s", len(prs), repo_id)
        return len(prs)
