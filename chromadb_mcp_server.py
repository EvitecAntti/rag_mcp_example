#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

from cli_helpers import DEFAULT_COLLECTION, DEFAULT_DB_DIR
from query_tool import format_context, simple_summary


mcp = FastMCP("chromadb-code-rag")
EMBEDDINGS = embedding_functions.DefaultEmbeddingFunction()


def _resolve_db_dir(db_dir: str) -> Path:
    return Path(db_dir).expanduser().resolve()


def _get_collection(db_dir: str, collection_name: str) -> Collection:
    path = _resolve_db_dir(db_dir)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=EMBEDDINGS,
    )


def _list_collections_sync(db_dir: str) -> List[Dict[str, str]]:
    path = _resolve_db_dir(db_dir)
    if not path.exists():
        return []
    client = chromadb.PersistentClient(path=str(path))
    data: List[Dict[str, str]] = []
    for collection in client.list_collections():
        try:
            count = collection.count()
        except Exception:
            count = 0
        entry = {
            "name": collection.name,
            "count": str(count),
        }
        if collection.metadata:
            entry["metadata"] = ", ".join(
                f"{key}={value}" for key, value in collection.metadata.items()
            )
        data.append(entry)
    return data


def _query_collection_sync(
    question: str,
    db_dir: str,
    collection_name: str,
    top_k: int,
) -> List[Tuple[Dict[str, str], str, float]]:
    collection = _get_collection(db_dir, collection_name)
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    metadatas: Sequence[Dict[str, str]] = results.get("metadatas", [[]])[0] or []
    documents: Sequence[str] = results.get("documents", [[]])[0] or []
    distances: Sequence[float] = results.get("distances", [[]])[0] or []

    contexts: List[Tuple[Dict[str, str], str, float]] = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        clean_metadata = {key: str(value) for key, value in (metadata or {}).items() if value is not None}
        contexts.append((clean_metadata, document or "", float(distance) if distance is not None else 0.0))
    return contexts


def _list_rows_sync(
    db_dir: str,
    collection_name: str,
    limit: int,
    offset: int,
    include_docs: bool,
) -> Tuple[int, List[Tuple[str, Dict[str, str], str | None]]]:
    client = chromadb.PersistentClient(path=str(_resolve_db_dir(db_dir)))
    collection = client.get_collection(name=collection_name)
    total = collection.count()
    include: List[str] = ["metadatas"]
    if include_docs:
        include.append("documents")
    data = collection.get(limit=limit, offset=offset, include=include)
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", []) if include_docs else []
    rows: List[Tuple[str, Dict[str, str], str | None]] = []
    for idx, row_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        clean_metadata = {key: str(value) for key, value in (metadata or {}).items() if value is not None}
        document = None
        if include_docs and idx < len(documents):
            document = documents[idx]
        rows.append((row_id, clean_metadata, document))
    return total, rows


def _format_rows_output(
    db_dir: Path,
    collection_name: str,
    total: int,
    offset: int,
    rows: List[Tuple[str, Dict[str, str], str | None]],
    include_docs: bool,
) -> str:
    lines = [
        f"Collection '{collection_name}' @ {db_dir}",
        f"Total rows: {total}",
        "",
    ]
    if not rows:
        lines.append("No rows found for the requested window.")
        return "\n".join(lines)
    for idx, (row_id, metadata, document) in enumerate(rows, start=offset + 1):
        lines.append(f"[{idx}] id={row_id}")
        if metadata:
            for key in sorted(metadata):
                lines.append(f"    {key}: {metadata[key]}")
        else:
            lines.append("    (no metadata)")
        if include_docs and document:
            snippet = document.strip()
            if len(snippet) > 700:
                snippet = snippet[:700] + "..."
            lines.append("    --- document ---")
            for doc_line in snippet.splitlines():
                lines.append(f"    {doc_line}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


@mcp.tool()
async def list_collections(db_dir: str = DEFAULT_DB_DIR) -> str:
    """List available Chroma collections in the configured database directory."""

    collections = await asyncio.to_thread(_list_collections_sync, db_dir)
    path = _resolve_db_dir(db_dir)
    if not collections:
        return f"No collections found in {path}. Ingest code with chromadb_code_rag.py first."
    lines = [f"Collections in {path}:"]
    for entry in collections:
        metadata = f" [{entry['metadata']}]" if "metadata" in entry else ""
        lines.append(f"- {entry['name']} ({entry['count']} rows){metadata}")
    return "\n".join(lines)


@mcp.tool()
async def query_codebase(
    question: str,
    top_k: int = 5,
    collection: str = DEFAULT_COLLECTION,
    db_dir: str = DEFAULT_DB_DIR,
) -> str:
    """Retrieve the most relevant code chunks for a natural-language question."""

    _validate_positive("top_k", top_k)
    try:
        contexts = await asyncio.to_thread(
            _query_collection_sync,
            question,
            db_dir,
            collection,
            top_k,
        )
    except Exception as exc:
        return f"Query failed: {exc}"

    if not contexts:
        return "No matches found. Make sure the collection has been ingested."

    snippets = []
    for idx, (metadata, document, distance) in enumerate(contexts, start=1):
        snippets.append(format_context(metadata, document, distance, idx))
    summary = simple_summary(question, [(metadata, document) for metadata, document, _ in contexts])
    header = f"Collection '{collection}' @ {_resolve_db_dir(db_dir)}"
    return "\n\n".join([header, *snippets, "-----------", summary])


@mcp.tool()
async def list_rows(
    collection: str = DEFAULT_COLLECTION,
    db_dir: str = DEFAULT_DB_DIR,
    limit: int = 10,
    offset: int = 0,
    include_documents: bool = False,
) -> str:
    """Inspect raw rows stored inside a Chroma collection."""

    _validate_positive("limit", limit)
    if offset < 0:
        raise ValueError("offset must be zero or greater.")
    try:
        total, rows = await asyncio.to_thread(
            _list_rows_sync,
            db_dir,
            collection,
            limit,
            offset,
            include_documents,
        )
    except chromadb.errors.InvalidCollectionException:
        return (
            f"Collection '{collection}' was not found in {_resolve_db_dir(db_dir)}. "
            "Run the ingestion script first."
        )
    except Exception as exc:
        return f"Failed to fetch rows: {exc}"

    output = _format_rows_output(
        _resolve_db_dir(db_dir),
        collection,
        total,
        offset,
        rows,
        include_documents,
    )
    return output


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
