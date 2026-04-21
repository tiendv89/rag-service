"""
Maps filesystem paths to their RAG source_type and determines feature_id.

Supported mappings (from technical design):

  workflow/workflow_skills/*/SKILL.md    → skill
  workflow/technical_skills/*/SKILL.md  → skill
  docs/features/*/product-spec.md       → product_spec
  docs/features/*/technical-design.md   → technical_design
  docs/features/*/tasks.md              → excluded (machine-generated, redundant)
  docs/**/*.md (other)                  → doc
  agents/<id>/log.jsonl                 → task_log
  CLAUDE.md, CLAUDE.shared.md           → claude_md
  README.md (top-level per repo)        → readme

Not indexed: source code, node_modules, vendor/, binaries, .env files.
"""

import os
import re
from typing import Optional

# Pattern → (source_type, feature_id_group or None)
# Patterns are matched against the path relative to the repo root.
# Order matters: more-specific patterns must precede broader ones.
_PATTERNS: list[tuple[re.Pattern, str, Optional[int]]] = [
    # workflow skills
    (re.compile(r"^workflow/workflow_skills/[^/]+/SKILL\.md$"), "skill", None),
    (re.compile(r"^workflow/technical_skills/[^/]+/SKILL\.md$"), "skill", None),
    # feature docs — explicit matches must appear before the generic doc pattern
    (re.compile(r"^docs/features/([^/]+)/product-spec\.md$"), "product_spec", 1),
    (re.compile(r"^docs/features/([^/]+)/technical-design\.md$"), "technical_design", 1),
    # docs folder — all .md files under docs/ not already matched above.
    # Captures feature_id from docs/features/<id>/... paths; None for all other docs paths.
    (re.compile(r"^docs/(?:features/([^/]+)/)?.*\.md$"), "doc", 1),
    # task logs — no feature_id association
    (re.compile(r"^agents/[^/]+/log\.jsonl$"), "task_log", None),
    # claude_md files at root or any level
    (re.compile(r"(?:^|/)CLAUDE(?:\.shared)?\.md$"), "claude_md", None),
    # top-level README per repo
    (re.compile(r"^README\.md$"), "readme", None),
]

# Paths that must never be indexed
_EXCLUDE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(^|/)node_modules(/|$)"),
    re.compile(r"(^|/)vendor(/|$)"),
    re.compile(r"(^|/)\.env($|\.)"),
    re.compile(r"\.pyc$"),
    re.compile(r"\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|bin|exe)$"),
    # Machine-generated planning docs — content covered by task YAML task_log entries
    re.compile(r"^docs/features/[^/]+/tasks\.md$"),
]


def classify_path(rel_path: str) -> Optional[tuple[str, Optional[str]]]:
    """
    Return (source_type, feature_id) for rel_path, or None if not indexed.

    rel_path must be a POSIX-style path relative to the repo root.
    feature_id is None for source_types that are not feature-scoped.
    """
    # Normalise separators
    rel_path = rel_path.replace(os.sep, "/")

    # Check exclusion patterns first
    for excl in _EXCLUDE_PATTERNS:
        if excl.search(rel_path):
            return None

    for pattern, source_type, feature_group in _PATTERNS:
        m = pattern.search(rel_path)
        if m:
            feature_id = m.group(feature_group) if feature_group else None
            return source_type, feature_id

    return None
