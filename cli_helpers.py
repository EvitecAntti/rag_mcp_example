from __future__ import annotations

import argparse
from typing import Sequence


DEFAULT_DB_DIR = ".chromadb"
DEFAULT_COLLECTION = "code-rag"
DEFAULT_EXTENSIONS = [
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".cs",
    ".cpp",
    ".c",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".kt",
    ".swift",
    ".scala",
    ".m",
    ".mm",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
]


def positive_int(value: str) -> int:
    """Argparse helper that enforces strictly positive integer inputs."""

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return ivalue


def parse_extensions(raw: str | None) -> Sequence[str]:
    """Normalize a comma-separated extensions string into dot-prefixed tokens."""

    if not raw:
        return DEFAULT_EXTENSIONS
    parts = [p.strip().lower() for p in raw.split(",")]
    normalized = []
    for part in parts:
        if not part:
            continue
        normalized.append(part if part.startswith(".") else f".{part}")
    return normalized


def build_ingest_parser() -> argparse.ArgumentParser:
    """Configure arguments for the ingestion script."""

    parser = argparse.ArgumentParser(
        description="Index a directory of code into a persistent ChromaDB collection."
    )
    parser.add_argument(
        "source_dir",
        help="Path to the directory you want to index.",
    )
    parser.add_argument(
        "--db-dir",
        default=DEFAULT_DB_DIR,
        help=f"Directory where the ChromaDB database will persist (default: {DEFAULT_DB_DIR}).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name inside ChromaDB (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--extensions",
        help="Comma-separated list of file extensions to include (default: common code files).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include files inside dot-directories.",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=2.0,
        help="Skip files larger than this size in megabytes (default: 2).",
    )
    parser.add_argument(
        "--chunk-lines",
        type=positive_int,
        default=40,
        help="Number of lines per chunk (default: 40).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=10,
        help="Line overlap between consecutive chunks (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=64,
        help="How many chunks to upsert per call (default: 64).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before ingesting.",
    )

    return parser


def build_query_parser() -> argparse.ArgumentParser:
    """Configure arguments for the query script."""

    parser = argparse.ArgumentParser(
        description="Query an existing ChromaDB code collection and optionally call a remote Ollama server."
    )
    parser.add_argument(
        "question",
        help="Natural-language question to search for.",
    )
    parser.add_argument(
        "--db-dir",
        default=DEFAULT_DB_DIR,
        help=f"Directory where the ChromaDB database will persist (default: {DEFAULT_DB_DIR}).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name inside ChromaDB (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=5,
        help="How many matches to retrieve (default: 5).",
    )
    parser.add_argument(
        "--ollama-model",
        help="Optional Ollama model name (e.g., llama3). If omitted, no LLM call is made.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL of the Ollama server (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-max-tokens",
        type=positive_int,
        default=400,
        help="Maximum tokens (num_predict) to request from Ollama (default: 400).",
    )

    return parser
