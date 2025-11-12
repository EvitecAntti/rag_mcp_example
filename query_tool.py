#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import chromadb
from chromadb.utils import embedding_functions
import requests

from cli_helpers import build_query_parser


def format_context(metadata: Dict[str, str], document: str, distance: float | None, idx: int) -> str:
    """Pretty-print a retrieved chunk with filename, line range, and distance."""

    header = f"[{idx}] {metadata.get('path')}:{metadata.get('start_line')}-{metadata.get('end_line')}"
    if distance is not None:
        header += f" (distance={distance:.4f})"
    snippet = textwrap.indent(document.strip(), prefix="    ")
    return f"{header}\n{snippet}"


def simple_summary(question: str, contexts: List[Tuple[Dict[str, str], str]]) -> str:
    """Offline fallback that lists retrieved snippets instead of calling an LLM."""

    if not contexts:
        return "No matching context found; try ingesting more files or broadening your query."
    bullets = []
    for idx, (metadata, _) in enumerate(contexts, start=1):
        bullets.append(
            f"{idx}. {metadata.get('path')} lines {metadata.get('start_line')}-{metadata.get('end_line')}"
        )
    bullet_text = "\n".join(bullets)
    template = textwrap.dedent(
        f"""
        No external LLM selected, so here are the most relevant chunks to help you answer manually.

        Question: {question}
        Ranked matches:
        {bullet_text}
        """
    ).strip()
    return template


def _normalize_base_url(url: str) -> str:
    """Ensure the Ollama base URL has a scheme and no trailing slash."""

    url = url.strip()
    if not url:
        return "http://localhost:11434"
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


def maybe_generate_with_ollama(
    question: str,
    contexts: List[Tuple[Dict[str, str], str]],
    model: str,
    base_url: str,
    max_tokens: int,
) -> str:
    """Send retrieved context to a remote Ollama server for answer synthesis."""

    context_blob = "\n\n".join(
        f"[{idx+1}] {metadata.get('path')}:{metadata.get('start_line')}-{metadata.get('end_line')}\n{document}"
        for idx, (metadata, document) in enumerate(contexts)
    )
    prompt = textwrap.dedent(
        f"""
        You are a concise assistant that answers questions about source code using only the provided context.
        Context:
        {context_blob}

        Question: {question}
        Answer using the context above in a few sentences. If the context does not contain the answer, say so.
        """
    ).strip()

    base = _normalize_base_url(base_url)
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to contact Ollama at {url}: {exc}") from exc

    data = response.json()
    answer = data.get("response", "").strip()
    return answer or "Ollama returned an empty response."


def handle_query(args: argparse.Namespace) -> None:
    """Embed the question, retrieve similar chunks, and display the results."""

    db_dir = Path(args.db_dir).expanduser().resolve()
    client = chromadb.PersistentClient(path=str(db_dir))
    default_embeddings = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=default_embeddings,
    )

    query = args.question
    results = collection.query(
        query_texts=[query],
        n_results=args.top_k,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    contexts = list(zip(metadatas, documents))

    if not documents:
        print("No matches found.")
        return

    for idx, (metadata, document, distance) in enumerate(
        zip(metadatas, documents, distances), start=1
    ):
        print(format_context(metadata, document, distance, idx))
        print()

    if args.ollama_model:
        answer = maybe_generate_with_ollama(
            question=query,
            contexts=contexts,
            model=args.ollama_model,
            base_url=args.ollama_url,
            max_tokens=args.ollama_max_tokens,
        )
    else:
        answer = simple_summary(query, contexts)
    print("-----------")
    print(answer)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the standalone query CLI."""

    parser = build_query_parser()
    args = parser.parse_args(argv)
    handle_query(args)


if __name__ == "__main__":
    main()
