"""Unit tests for services/shared/schema.py."""

import pytest

from services.shared.schema import (
    VALID_SOURCE_TYPES,
    ChunkPayload,
    build_workspace_filter,
)


class TestChunkPayload:
    """Tests for the ChunkPayload dataclass validation."""

    def test_valid_payload(self):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type="skill",
            source_path="workflow/workflow_skills/start-implementation/SKILL.md",
            chunk_index=0,
            indexed_at="2026-04-19T00:00:00+00:00",
        )
        assert p.workspace_id == "workspace"
        assert p.feature_id is None

    def test_valid_payload_with_feature_id(self):
        p = ChunkPayload(
            workspace_id="faro",
            source_type="technical_design",
            source_path="docs/features/agent-rag-mcp/technical-design.md",
            chunk_index=2,
            indexed_at="2026-04-19T00:00:00+00:00",
            feature_id="agent-rag-mcp",
        )
        assert p.feature_id == "agent-rag-mcp"

    def test_missing_workspace_id_raises(self):
        with pytest.raises(ValueError, match="workspace_id is required"):
            ChunkPayload(
                workspace_id="",
                source_type="skill",
                source_path="some/path.md",
                chunk_index=0,
                indexed_at="2026-04-19T00:00:00+00:00",
            )

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValueError, match="source_type must be one of"):
            ChunkPayload(
                workspace_id="workspace",
                source_type="unknown_type",
                source_path="some/path.md",
                chunk_index=0,
                indexed_at="2026-04-19T00:00:00+00:00",
            )

    def test_empty_source_path_raises(self):
        with pytest.raises(ValueError, match="source_path is required"):
            ChunkPayload(
                workspace_id="workspace",
                source_type="readme",
                source_path="",
                chunk_index=0,
                indexed_at="2026-04-19T00:00:00+00:00",
            )

    def test_negative_chunk_index_raises(self):
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            ChunkPayload(
                workspace_id="workspace",
                source_type="readme",
                source_path="README.md",
                chunk_index=-1,
                indexed_at="2026-04-19T00:00:00+00:00",
            )

    @pytest.mark.parametrize("source_type", sorted(VALID_SOURCE_TYPES))
    def test_all_valid_source_types_accepted(self, source_type):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type=source_type,
            source_path="some/path.md",
            chunk_index=0,
            indexed_at="2026-04-19T00:00:00+00:00",
        )
        assert p.source_type == source_type

    def test_doc_source_type_valid(self):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type="doc",
            source_path="docs/architecture/overview.md",
            chunk_index=0,
            indexed_at="2026-04-19T00:00:00+00:00",
        )
        assert p.source_type == "doc"

    def test_source_code_source_type_valid(self):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type="source_code",
            source_path="services/indexer/chunker.py",
            chunk_index=0,
            indexed_at="2026-04-19T00:00:00+00:00",
        )
        assert p.source_type == "source_code"

    @pytest.mark.parametrize("source_type", ["skill", "task_log", "product_spec", "technical_design", "readme", "claude_md"])
    def test_legacy_source_types_still_valid(self, source_type):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type=source_type,
            source_path="some/path.md",
            chunk_index=0,
            indexed_at="2026-04-19T00:00:00+00:00",
        )
        assert p.source_type == source_type

    def test_to_dict_contains_required_keys(self):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type="skill",
            source_path="some/path.md",
            chunk_index=1,
            indexed_at="2026-04-19T00:00:00+00:00",
            feature_id="agent-rag-mcp",
        )
        d = p.to_dict()
        assert d["workspace_id"] == "workspace"
        assert d["source_type"] == "skill"
        assert d["source_path"] == "some/path.md"
        assert d["feature_id"] == "agent-rag-mcp"
        assert d["chunk_index"] == 1
        assert d["indexed_at"] == "2026-04-19T00:00:00+00:00"

    def test_to_dict_feature_id_none_when_not_provided(self):
        p = ChunkPayload(
            workspace_id="workspace",
            source_type="readme",
            source_path="README.md",
            chunk_index=0,
            indexed_at="2026-04-19T00:00:00+00:00",
        )
        d = p.to_dict()
        assert d["feature_id"] is None

    def test_now_iso_returns_string(self):
        ts = ChunkPayload.now_iso()
        assert isinstance(ts, str)
        assert "T" in ts


class TestBuildWorkspaceFilter:
    """Tests for build_workspace_filter."""

    def test_returns_correct_must_filter(self):
        f = build_workspace_filter("workspace")
        assert f == {
            "must": [
                {"key": "workspace_id", "match": {"value": "workspace"}}
            ]
        }

    def test_different_workspace_id(self):
        f = build_workspace_filter("faro")
        assert f["must"][0]["match"]["value"] == "faro"

    def test_empty_workspace_id_raises(self):
        with pytest.raises(ValueError, match="workspace_id is required"):
            build_workspace_filter("")

    def test_none_workspace_id_raises(self):
        with pytest.raises((ValueError, TypeError)):
            build_workspace_filter(None)  # type: ignore[arg-type]

    def test_filter_structure_has_must_key(self):
        f = build_workspace_filter("workspace")
        assert "must" in f
        assert isinstance(f["must"], list)
        assert len(f["must"]) == 1

    def test_filter_must_entry_has_key_and_match(self):
        f = build_workspace_filter("workspace")
        entry = f["must"][0]
        assert entry["key"] == "workspace_id"
        assert "match" in entry
        assert "value" in entry["match"]
