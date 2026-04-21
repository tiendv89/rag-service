"""Unit tests for services/indexer/source_mapper.py."""

import pytest

from services.indexer.source_mapper import classify_path


class TestClassifyPath:
    # ------------------------------------------------------------------
    # Skill paths
    # ------------------------------------------------------------------
    def test_workflow_skill(self):
        result = classify_path("workflow/workflow_skills/start-implementation/SKILL.md")
        assert result == ("skill", None)

    def test_technical_skill(self):
        result = classify_path("workflow/technical_skills/python-best-practices/SKILL.md")
        assert result == ("skill", None)

    # ------------------------------------------------------------------
    # Feature docs
    # ------------------------------------------------------------------
    def test_product_spec(self):
        result = classify_path("docs/features/agent-rag-mcp/product-spec.md")
        assert result == ("product_spec", "agent-rag-mcp")

    def test_technical_design(self):
        result = classify_path("docs/features/agent-rag-mcp/technical-design.md")
        assert result == ("technical_design", "agent-rag-mcp")

    def test_feature_id_extracted_correctly(self):
        result = classify_path("docs/features/my-feature-id/product-spec.md")
        assert result is not None
        source_type, feature_id = result
        assert feature_id == "my-feature-id"

    # ------------------------------------------------------------------
    # Task logs
    # ------------------------------------------------------------------
    def test_task_log(self):
        result = classify_path("agents/bot-01/log.jsonl")
        assert result == ("task_log", None)

    # ------------------------------------------------------------------
    # Claude MD
    # ------------------------------------------------------------------
    def test_claude_md_root(self):
        result = classify_path("CLAUDE.md")
        assert result == ("claude_md", None)

    def test_claude_shared_md(self):
        result = classify_path("CLAUDE.shared.md")
        assert result == ("claude_md", None)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    def test_top_level_readme(self):
        result = classify_path("README.md")
        assert result == ("readme", None)

    # ------------------------------------------------------------------
    # Excluded paths
    # ------------------------------------------------------------------
    def test_node_modules_excluded(self):
        assert classify_path("node_modules/package/index.js") is None

    def test_vendor_excluded(self):
        assert classify_path("vendor/lib/something.py") is None

    def test_env_file_excluded(self):
        assert classify_path(".env") is None
        assert classify_path(".env.local") is None

    def test_pyc_excluded(self):
        assert classify_path("services/shared/schema.pyc") is None

    def test_binary_image_excluded(self):
        assert classify_path("docs/image.png") is None
        assert classify_path("assets/logo.jpg") is None

    # ------------------------------------------------------------------
    # Doc paths — inclusion
    # ------------------------------------------------------------------
    def test_docs_non_feature_path_is_doc(self):
        result = classify_path("docs/architecture/overview.md")
        assert result == ("doc", None)

    def test_docs_top_level_file_is_doc(self):
        result = classify_path("docs/random-notes.md")
        assert result == ("doc", None)

    def test_docs_feature_guide_is_doc_with_feature_id(self):
        result = classify_path("docs/features/my-feat/guide.md")
        assert result == ("doc", "my-feat")

    def test_docs_feature_nested_is_doc_with_feature_id(self):
        result = classify_path("docs/features/agent-rag-v2/adr/001-chunking.md")
        assert result == ("doc", "agent-rag-v2")

    # ------------------------------------------------------------------
    # Doc paths — exclusion (already covered by specific source types or excluded)
    # ------------------------------------------------------------------
    def test_docs_product_spec_is_not_doc(self):
        result = classify_path("docs/features/x/product-spec.md")
        assert result == ("product_spec", "x")

    def test_docs_technical_design_is_not_doc(self):
        result = classify_path("docs/features/x/technical-design.md")
        assert result == ("technical_design", "x")

    def test_docs_tasks_md_not_indexed(self):
        assert classify_path("docs/features/x/tasks.md") is None

    # ------------------------------------------------------------------
    # Source code inclusion
    # ------------------------------------------------------------------
    def test_python_file_indexed(self):
        assert classify_path("services/shared/schema.py") == ("source_code", None)

    def test_typescript_file_indexed(self):
        assert classify_path("src/components/Button.ts") == ("source_code", None)

    def test_tsx_file_indexed(self):
        assert classify_path("src/pages/Index.tsx") == ("source_code", None)

    def test_js_file_indexed(self):
        assert classify_path("scripts/deploy.js") == ("source_code", None)

    def test_go_file_indexed(self):
        assert classify_path("cmd/server/main.go") == ("source_code", None)

    def test_test_file_included(self):
        # Test files are explicitly included — they document real usage patterns
        assert classify_path("services/auth.test.ts") == ("source_code", None)

    def test_spec_file_included(self):
        assert classify_path("tests/auth.spec.ts") == ("source_code", None)

    def test_python_test_file_included(self):
        assert classify_path("tests/test_auth.py") == ("source_code", None)

    # ------------------------------------------------------------------
    # Source code exclusions
    # ------------------------------------------------------------------
    def test_node_modules_ts_excluded(self):
        assert classify_path("node_modules/foo.ts") is None

    def test_pycache_excluded(self):
        assert classify_path("services/__pycache__/schema.cpython-311.pyc") is None
        assert classify_path("__pycache__/module.py") is None

    def test_dist_excluded(self):
        assert classify_path("dist/bundle.js") is None

    def test_build_excluded(self):
        assert classify_path("build/output.js") is None

    def test_next_excluded(self):
        assert classify_path(".next/server/pages/index.js") is None

    def test_out_excluded(self):
        assert classify_path("out/static/chunk.js") is None

    def test_migrations_excluded(self):
        assert classify_path("migrations/001_init.py") is None

    # ------------------------------------------------------------------
    # Unmatched paths return None
    # ------------------------------------------------------------------
    def test_nested_readme_not_indexed(self):
        # Only top-level README.md is indexed
        assert classify_path("subdir/README.md") is None
