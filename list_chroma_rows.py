#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import chromadb

from cli_helpers import DEFAULT_COLLECTION, DEFAULT_DB_DIR


def build_parser() -> argparse.ArgumentParser:
    """Build CLI for listing stored rows in a Chroma collection."""

    parser = argparse.ArgumentParser(
        description="Inspect the rows/documents stored inside a ChromaDB collection."
    )
    parser.add_argument(
        "--db-dir",
        default=DEFAULT_DB_DIR,
        help=f"Directory where the ChromaDB database lives (default: {DEFAULT_DB_DIR}).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection name to inspect (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to fetch (default: 20).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of rows to skip before listing (default: 0).",
    )
    parser.add_argument(
        "--show-docs",
        action="store_true",
        help="Include document text in the output (default: metadata only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump raw JSON instead of a human-readable table.",
    )
    return parser


def ensure_collection(db_dir: Path, collection_name: str):
    """Open the requested collection, raising a helpful error if missing."""

    db_dir = db_dir.expanduser().resolve()
    if not db_dir.exists():
        raise SystemExit(f"Database directory '{db_dir}' does not exist.")
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        return client.get_collection(name=collection_name)
    except chromadb.errors.InvalidCollectionException as exc:
        raise SystemExit(f"Collection '{collection_name}' was not found in {db_dir}.") from exc


def format_row(idx: int, row_id: str, metadata: dict, document: str | None) -> str:
    """Format a single row for console output."""

    lines = [f"[{idx}] id={row_id}"]
    if metadata:
        for key, value in metadata.items():
            lines.append(f"    {key}: {value}")
    else:
        lines.append("    (no metadata)")
    if document is not None:
        snippet = document.strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + " …"
        lines.append("    --- document ---")
        lines.append("    " + snippet.replace("\n", "\n    "))
    return "\n".join(lines)


def list_rows(args: argparse.Namespace) -> None:
    """Fetch and display rows from the specified Chroma collection."""

    collection = ensure_collection(Path(args.db_dir), args.collection)
    total = collection.count()
    include: Sequence[str] = ["metadatas"]
    if args.show_docs:
        include = ["metadatas", "documents"]
    try:
        data = collection.get(limit=args.limit, offset=args.offset, include=list(include))
    except ValueError as exc:
        raise SystemExit(f"Failed to fetch rows: {exc}") from exc

    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])

    if args.json:
        payload = {
            "count": len(ids),
            "offset": args.offset,
            "total": total,
            "ids": ids,
            "metadatas": metadatas,
            "documents": documents if args.show_docs else None,
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Collection: {args.collection} (total rows: {total})")
    if not ids:
        print("No rows found for the given offset/limit.")
        return

    for rel_idx, row_id in enumerate(ids):
        idx = args.offset + rel_idx + 1
        metadata = metadatas[rel_idx] if metadatas else {}
        document = documents[rel_idx] if args.show_docs and documents else None
        print(format_row(idx, row_id, metadata, document))
        print("-" * 40)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    list_rows(args)


if __name__ == "__main__":
    main()
