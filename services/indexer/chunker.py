"""
Per-source-type document chunking logic.

Chunking strategies:
- skill, claude_md:      Whole-file (single chunk per document)
- product_spec,
  technical_design,
  readme:                512 tokens / 50-token overlap (sliding window)
- task_log:              One chunk per JSONL log entry
- source_code:           AST-aware (tree-sitter), fallback to sliding window

Token counting uses a simple whitespace-based word approximation
(4 chars ≈ 1 token) suitable for the BAAI/bge-base-en-v1.5 input budget.
"""

import json
import logging
import os
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


# Maximum node size before splitting with sliding window (~1500 tokens)
_MAX_NODE_CHARS = 6000

# File extension → tree-sitter language name
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".go": "go",
}

# Top-level node types representing standalone functions/methods
_FUNCTION_NODE_TYPES = frozenset({
    "function_definition",   # Python
    "function_declaration",  # TypeScript, Go
    "method_declaration",    # Go receiver methods
})

# Class/struct node types
_CLASS_NODE_TYPES = frozenset({
    "class_definition",   # Python
    "class_declaration",  # TypeScript
})

# Method node types inside a class body
_METHOD_NODE_TYPES = frozenset({
    "function_definition",  # Python method inside class block
    "method_definition",    # TypeScript method inside class_body
})


def _infer_language(source_path: str) -> Optional[str]:
    ext = os.path.splitext(source_path)[1].lower()
    return _EXT_TO_LANGUAGE.get(ext)


def _get_parser(language: str):
    """Return a tree-sitter Parser for the given language, or None on failure."""
    try:
        from tree_sitter import Language, Parser
        if language == "python":
            import tree_sitter_python as tsp
            return Parser(Language(tsp.language()))
        elif language in ("typescript", "javascript"):
            import tree_sitter_typescript as tst
            return Parser(Language(tst.language_typescript()))
        elif language == "go":
            import tree_sitter_go as tsg
            return Parser(Language(tsg.language()))
    except Exception as exc:
        logger.debug("Failed to create parser for %s: %s", language, exc)
    return None


def _extract_node_name(node) -> Optional[str]:
    """Return the name identifier text of a function or class node."""
    for child in node.children:
        if child.type in (
            "identifier", "type_identifier",
            "field_identifier", "property_identifier",
        ):
            return child.text.decode("utf-8", errors="replace")
    return None


def _get_class_body(node):
    """Return the body/block child of a class node, or None."""
    for child in node.children:
        if child.type in ("block", "class_body", "declaration_list"):
            return child
    return None


def _emit_node(prefix: str, node_text: str, chunks: list[str]) -> None:
    """Append node_text (with prefix) to chunks, splitting if it exceeds _MAX_NODE_CHARS."""
    if len(node_text) <= _MAX_NODE_CHARS:
        chunks.append(prefix + node_text)
        return
    # For large nodes, prefix each sub-chunk with the function signature line
    signature_line = node_text.split("\n", 1)[0]
    for sub in _sliding_window_chunks(node_text):
        chunks.append(prefix + signature_line + "\n" + sub)


def _ast_chunk_source(content: str, language: str, source_path: str) -> list[str]:
    """
    Chunk source code using tree-sitter AST parsing.

    Walks top-level function/class nodes. For classes, emits each method
    individually prefixed with ``# class: <ClassName>``.

    Falls back to ``_sliding_window_chunks`` when:
    - the parser cannot be created (unsupported language)
    - parsing raises an exception
    - zero nodes are extracted from the tree
    """
    parser = _get_parser(language)
    if parser is None:
        return _sliding_window_chunks(content)

    try:
        content_bytes = content.encode("utf-8", errors="replace")
        tree = parser.parse(content_bytes)
        chunks: list[str] = []

        for node in tree.root_node.children:
            # Standalone functions / top-level declarations
            if node.type in _FUNCTION_NODE_TYPES:
                node_text = content_bytes[node.start_byte:node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                _emit_node(f"# file: {source_path}\n", node_text, chunks)

            # Classes — emit each method individually
            elif node.type in _CLASS_NODE_TYPES:
                class_name = _extract_node_name(node)
                class_body = _get_class_body(node)
                if class_body:
                    methods_emitted = 0
                    for child in class_body.children:
                        if child.type in _METHOD_NODE_TYPES:
                            method_text = content_bytes[child.start_byte:child.end_byte].decode(
                                "utf-8", errors="replace"
                            )
                            prefix = f"# file: {source_path}\n"
                            if class_name:
                                prefix += f"# class: {class_name}\n"
                            _emit_node(prefix, method_text, chunks)
                            methods_emitted += 1
                    if methods_emitted == 0:
                        # Class with no extractable methods — emit the whole class
                        node_text = content_bytes[node.start_byte:node.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                        _emit_node(f"# file: {source_path}\n", node_text, chunks)

            # Arrow functions assigned to a variable: const foo = () => { ... }
            elif node.type in ("lexical_declaration", "variable_declaration"):
                for declarator in node.children:
                    if declarator.type == "variable_declarator":
                        for child in declarator.children:
                            if child.type == "arrow_function":
                                node_text = content_bytes[node.start_byte:node.end_byte].decode(
                                    "utf-8", errors="replace"
                                )
                                _emit_node(f"# file: {source_path}\n", node_text, chunks)
                                break

        if not chunks:
            return _sliding_window_chunks(content)

        return chunks

    except Exception as exc:
        logger.warning("tree-sitter parsing failed for %s: %s", source_path, exc)
        return _sliding_window_chunks(content)


def chunk_document(source_type: str, content: str, source_path: str = "") -> list[str]:
    """
    Chunk a document according to its source_type.

    Returns a list of string chunks. Each chunk will become one Qdrant point.

    - skill, claude_md:                        single-element list (whole file)
    - product_spec, technical_design, readme,
      doc:                                     sliding-window chunks (512 tok / 50 overlap)
    - task_log:                                one element per valid JSONL line
    - source_code:                             AST-aware chunking via tree-sitter;
                                               source_path required for language inference;
                                               falls back to sliding window if language is
                                               unknown or parsing fails

    Raises ValueError for unknown source_types.
    """
    if source_type in ("skill", "claude_md"):
        text = content.strip()
        return [text] if text else []

    if source_type in ("product_spec", "technical_design", "readme", "doc"):
        return _sliding_window_chunks(content)

    if source_type == "task_log":
        return _chunk_task_log(content)

    if source_type == "source_code":
        language = _infer_language(source_path)
        if language:
            return _ast_chunk_source(content, language, source_path)
        return _sliding_window_chunks(content)

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
