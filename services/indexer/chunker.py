"""
Per-source-type document chunking logic.

Chunking strategies:
- skill, claude_md:      Whole-file (single chunk per document)
- product_spec,
  technical_design,
  readme:                512 tokens / 50-token overlap (sliding window)
- task_log:              One chunk per JSONL log entry

Source code files are not indexed — code queries are handled by GitNexus.

Token counting uses a simple whitespace-based word approximation
(4 chars ≈ 1 token) suitable for the BAAI/bge-base-en-v1.5 input budget.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Chunking parameters for sliding-window strategy
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50

# Approximate chars-per-token for split-based counting
_CHARS_PER_TOKEN = 4


def _approx_token_count(text: str) -> int:
    """Approximate token count: len(text) / 4."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _sliding_window_chunks(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_tokens tokens.

    The split is performed on word boundaries to avoid cutting mid-word.
    Returns at least one chunk even for very short texts.
    """
    words = text.split()
    if not words:
        return [""]

    chars_per_chunk = chunk_tokens * _CHARS_PER_TOKEN
    chars_per_overlap = overlap_tokens * _CHARS_PER_TOKEN

    chunks: list[str] = []
    start_char = 0
    text_len = len(text)

    while start_char < text_len:
        end_char = start_char + chars_per_chunk
        chunk = text[start_char:end_char]

        # Extend to the next word boundary unless we are at the end
        if end_char < text_len:
            next_space = text.find(" ", end_char)
            if next_space != -1:
                chunk = text[start_char:next_space]
            # If no space found, take to end
        else:
            chunk = text[start_char:]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end_char >= text_len:
            break

        # Advance start by (chunk_size - overlap)
        advance = chars_per_chunk - chars_per_overlap
        start_char += max(advance, 1)

    return chunks if chunks else [text.strip()]


def chunk_document(source_type: str, content: str, source_path: str = "") -> list[str]:
    """
    Chunk a document according to its source_type.

    Returns a list of string chunks. Each chunk will become one Qdrant point.

    - skill, claude_md:                        single-element list (whole file)
    - product_spec, technical_design, readme,
      doc:                                     sliding-window chunks (512 tok / 50 overlap)
    - task_log:                                one element per valid JSONL line
    Raises ValueError for unknown source_types.
    """
    if source_type in ("skill", "claude_md"):
        text = content.strip()
        return [text] if text else []

    if source_type in ("product_spec", "technical_design", "readme", "doc"):
        return _sliding_window_chunks(content)

    if source_type == "task_log":
        return _chunk_task_log(content)

    raise ValueError(f"Unknown source_type: {source_type!r}")


def _chunk_task_log(content: str) -> list[str]:
    """
    Split a JSONL task log into one chunk per log entry.

    Non-JSON lines are skipped with a warning.
    Returns an empty list if no valid entries are found.
    """
    chunks: list[str] = []
    for lineno, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            # Serialize back to a compact but readable string for embedding
            chunks.append(json.dumps(entry, ensure_ascii=False))
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON line %d in task log", lineno)
    return chunks
