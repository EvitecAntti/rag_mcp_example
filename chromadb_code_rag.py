#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import chromadb
from chromadb.utils import embedding_functions

from cli_helpers import build_ingest_parser, parse_extensions


@dataclass
class Chunk:
    """Small wrapper for a code chunk plus the metadata needed for retrieval."""

    doc_id: str
    document: str
    metadata: Dict[str, str]


def iter_code_files(
    root: Path,
    extensions: Sequence[str],
    include_hidden: bool,
    max_file_mb: float,
) -> Iterator[Path]:
    """Yield files under root that match extension, visibility, and size filters."""

    max_bytes = max_file_mb * 1024 * 1024
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if not include_hidden and any(part.startswith(".") for part in relative_parts):
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        yield path


def chunk_text(
    text: str,
    chunk_lines: int,
    chunk_overlap: int,
) -> Iterator[Tuple[str, int, int]]:
    """Split text into overlapping windows and report each chunk with line numbers."""
    lines = text.splitlines()
    if not lines:
        return
    step = max(1, chunk_lines - chunk_overlap)
    start = 0
    while start < len(lines):
        end = min(len(lines), start + chunk_lines)
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield chunk, start + 1, end
        start += step


def batched(iterable: Iterable[Chunk], size: int) -> Iterator[List[Chunk]]:
    """Group an iterable of chunks into fixed-size lists for bulk upserts."""

    batch: List[Chunk] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def discover_chunks(
    source_dir: Path,
    extensions: Sequence[str],
    include_hidden: bool,
    max_file_mb: float,
    chunk_lines: int,
    chunk_overlap: int,
    collection: str,
) -> Iterator[Chunk]:
    """Walk the source tree and yield Chunk objects for each processed file."""

    for path in iter_code_files(source_dir, extensions, include_hidden, max_file_mb):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        relative = path.relative_to(source_dir).as_posix()
        for idx, (chunk, start_line, end_line) in enumerate(
            chunk_text(text, chunk_lines, chunk_overlap)
        ):
            doc_id = f"{relative}::{idx}"
            metadata = {
                "path": relative,
                "start_line": str(start_line),
                "end_line": str(end_line),
                "source_dir": str(source_dir),
                "collection": collection,
            }
            yield Chunk(doc_id=doc_id, document=chunk, metadata=metadata)


def ensure_collection(
    db_dir: Path,
    collection_name: str,
    reset: bool,
) -> chromadb.api.models.Collection.Collection:
    """Create (and optionally reset) the persistent Chroma collection for this run."""

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'.")
        except chromadb.errors.InvalidCollectionException:
            pass
    default_embeddings = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_embeddings,
        metadata={"platform": "chromadb", "kind": "code-rag"},
    )


def handle_ingest(args: argparse.Namespace) -> None:
    """CLI handler that indexes a directory and writes chunks into ChromaDB."""

    source = Path(args.source_dir).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory '{source}' does not exist or is not a folder.")

    db_dir = Path(args.db_dir).expanduser().resolve()
    collection = ensure_collection(db_dir, args.collection, args.reset)

    extensions = parse_extensions(args.extensions)
    chunk_iter = discover_chunks(
        source_dir=source,
        extensions=extensions,
        include_hidden=args.include_hidden,
        max_file_mb=args.max_file_mb,
        chunk_lines=args.chunk_lines,
        chunk_overlap=args.chunk_overlap,
        collection=args.collection,
    )
    total = 0
    has_data = False
    for batch in batched(chunk_iter, args.batch_size):
        collection.upsert(
            ids=[c.doc_id for c in batch],
            documents=[c.document for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        total += len(batch)
        has_data = True
    if not has_data:
        print("No chunks were produced; check your filters.")
        return
    print(f"Ingested {total} chunks from {source}.")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ingestion CLI."""

    parser = build_ingest_parser()
    args = parser.parse_args(argv)
    handle_ingest(args)


if __name__ == "__main__":
    main()
