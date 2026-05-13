"""Unit tests for services/indexer/branch_parser.py."""

import pytest

from services.indexer.branch_parser import parse_branch


class TestParseBranch:
    def test_feature_branch_with_task_id(self):
        feature_id, task_id = parse_branch("feature/agent-rag-pr-index-T1")
        assert feature_id == "agent-rag-pr-index"
        assert task_id == "1"

    def test_feature_branch_without_task_id(self):
        feature_id, task_id = parse_branch("feature/agent-rag-pr-index")
        assert feature_id == "agent-rag-pr-index"
        assert task_id is None

    def test_non_feature_branch_returns_none(self):
        feature_id, task_id = parse_branch("main")
        assert feature_id is None
        assert task_id is None

    def test_empty_string_returns_none(self):
        feature_id, task_id = parse_branch("")
        assert feature_id is None
        assert task_id is None

    def test_multi_digit_task_id(self):
        feature_id, task_id = parse_branch("feature/my-feature-T12")
        assert feature_id == "my-feature"
        assert task_id == "12"

    def test_feature_branch_with_underscores(self):
        feature_id, task_id = parse_branch("feature/my_feature_name-T3")
        assert feature_id == "my_feature_name"
        assert task_id == "3"

    def test_complex_feature_id(self):
        feature_id, task_id = parse_branch("feature/agent-runtime-redesign-T5")
        assert feature_id == "agent-runtime-redesign"
        assert task_id == "5"

    def test_hotfix_branch_returns_none(self):
        feature_id, task_id = parse_branch("hotfix/security-patch")
        assert feature_id is None
        assert task_id is None

    def test_returns_tuple(self):
        result = parse_branch("feature/foo-T1")
        assert isinstance(result, tuple)
        assert len(result) == 2
