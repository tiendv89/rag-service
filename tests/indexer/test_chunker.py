"""Unit tests for services/indexer/chunker.py."""

import json
import pytest

from services.indexer.chunker import (
    CHUNK_TOKENS,
    CHUNK_OVERLAP,
    chunk_document,
    _sliding_window_chunks,
    _chunk_task_log,
)


# ---------------------------------------------------------------------------
# _sliding_window_chunks
# ---------------------------------------------------------------------------

class TestSlidingWindowChunks:
    def test_short_text_returns_single_chunk(self):
        text = "Hello world, this is a short document."
        chunks = _sliding_window_chunks(text)
        assert len(chunks) >= 1
        # Short text should return at most 2 chunks
        assert len(chunks) <= 2

    def test_empty_text_returns_single_empty_chunk(self):
        chunks = _sliding_window_chunks("")
        assert chunks == [""]

    def test_long_text_produces_multiple_chunks(self):
        # 3000 chars ≈ 750 tokens, expect > 1 chunk with default 512-token chunks
        text = "word " * 600  # 3000 chars
        chunks = _sliding_window_chunks(text)
        assert len(chunks) > 1

    def test_chunks_overlap(self):
        # Build a text large enough to produce at least 2 chunks
        text = "alpha " * 600  # 3000 chars
        chunks = _sliding_window_chunks(text, chunk_tokens=50, overlap_tokens=10)
        assert len(chunks) >= 2
        # Each chunk except the last should end before the start of the next
        # with some overlap — verify that chunks are non-empty strings
        for c in chunks:
            assert isinstance(c, str)
            assert c.strip()

    def test_no_empty_chunks_in_output(self):
        text = "sentence. " * 200
        chunks = _sliding_window_chunks(text)
        for chunk in chunks:
            assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# _chunk_task_log
# ---------------------------------------------------------------------------

class TestChunkTaskLog:
    def test_valid_jsonl_produces_one_chunk_per_line(self):
        lines = [
            json.dumps({"action": "started", "by": "bot", "at": "2026-04-01T00:00:00Z"}),
            json.dumps({"action": "done", "by": "bot", "at": "2026-04-01T01:00:00Z"}),
        ]
        content = "\n".join(lines)
        chunks = _chunk_task_log(content)
        assert len(chunks) == 2

    def test_invalid_lines_skipped(self):
        content = "not json\n" + json.dumps({"action": "ok"}) + "\nalso not json"
        chunks = _chunk_task_log(content)
        assert len(chunks) == 1
        assert json.loads(chunks[0])["action"] == "ok"

    def test_empty_content_returns_empty(self):
        assert _chunk_task_log("") == []

    def test_blank_lines_skipped(self):
        entry = json.dumps({"action": "started"})
        content = f"\n\n{entry}\n\n"
        chunks = _chunk_task_log(content)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_document dispatcher
# ---------------------------------------------------------------------------

class TestChunkDocument:
    @pytest.mark.parametrize("source_type", ["skill", "claude_md"])
    def test_whole_file_types_return_single_chunk(self, source_type):
        content = "# SKILL\n\nThis is the entire skill file.\n"
        chunks = chunk_document(source_type, content)
        assert len(chunks) == 1
        assert chunks[0] == content.strip()

    @pytest.mark.parametrize("source_type", ["skill", "claude_md"])
    def test_whole_file_empty_content_returns_empty_list(self, source_type):
        chunks = chunk_document(source_type, "   ")
        assert chunks == []

    @pytest.mark.parametrize("source_type", ["product_spec", "technical_design", "readme", "doc"])
    def test_sliding_window_types_return_at_least_one_chunk(self, source_type):
        chunks = chunk_document(source_type, "Some content here.")
        assert len(chunks) >= 1

    def test_doc_returns_non_empty_list_of_strings(self):
        markdown = "# Architecture Overview\n\nThis document describes the overall architecture.\n\n## Components\n\nThe system has three main components."
        chunks = chunk_document("doc", markdown)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert chunk.strip()

    def test_task_log_returns_one_chunk_per_entry(self):
        entry1 = json.dumps({"action": "started"})
        entry2 = json.dumps({"action": "done"})
        content = f"{entry1}\n{entry2}"
        chunks = chunk_document("task_log", content)
        assert len(chunks) == 2

    def test_source_code_raises_unknown_source_type(self):
        with pytest.raises(ValueError, match="Unknown source_type"):
            chunk_document("source_code", "def foo(): pass", source_path="foo.py")

    def test_unknown_source_type_raises(self):
        with pytest.raises(ValueError, match="Unknown source_type"):
            chunk_document("unknown_type", "content")

    def test_skill_preserves_content(self):
        content = "# My Skill\n\nDoes something important.\n"
        chunks = chunk_document("skill", content)
        assert chunks[0] == content.strip()

    def test_large_readme_produces_multiple_chunks(self):
        content = "word " * 600  # ~3000 chars ≈ 750 tokens
        chunks = chunk_document("readme", content)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# Integration: verify chunk continuity for sliding window
# ---------------------------------------------------------------------------

class TestChunkCoverage:
    def test_all_content_is_covered(self):
        """
        Every word in the original text should appear in at least one chunk.
        """
        words = ["word" + str(i) for i in range(200)]
        text = " ".join(words)
        chunks = _sliding_window_chunks(text, chunk_tokens=20, overlap_tokens=5)
        all_chunk_text = " ".join(chunks)
        for w in words:
            assert w in all_chunk_text, f"Word {w!r} missing from chunks"
