# ChromaDB Code RAG Helper

This repository now ships two tiny command-line helpers: one indexes a directory of source code into [ChromaDB](https://www.trychroma.com/), and the other queries that persisted collection using Retrieval-Augmented Generation (RAG) techniques. Everything stays local - files are chunked, embedded, and stored on disk - and you can optionally let a remote Ollama server synthesize natural-language answers from the retrieved context. Without an LLM the query tool still surfaces the most relevant chunks so you can answer manually.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate  # PowerShell on Windows
pip install -r requirements.txt
```

## Usage

There are two entry points:

- `chromadb_code_rag.py` - handles ingestion / indexing.
- `query_tool.py` - performs retrieval (and optional LLM synthesis).
- `list_chroma_rows.py` - inspects the raw rows/documents stored in a collection.

### 1. Ingest a directory

```bash
python chromadb_code_rag.py path\to\your\repo --reset
```

Important flags:

- `--db-dir`: where to store the persistent database (default `.chromadb/`).
- `--collection`: unique name for your dataset (default `code-rag`).
- `--extensions`: comma-separated list of file extensions to include. Defaults cover popular languages plus `json`, `yaml`, `toml`, `md`.
- `--chunk-lines` / `--chunk-overlap`: control chunk size and overlap in lines (defaults 40 / 10).
- `--max-file-mb`: skip very large files (default `2` MB).
- `--reset`: drop the existing collection before re-ingesting.

### 2. Ask a question

```bash
python query_tool.py "How is authentication handled?"
```

Flags:

- `--top-k`: number of chunks to retrieve (default `5`).
- `--ollama-model`: when provided (e.g., `llama3`) the script sends the retrieved context to your Ollama server for answer synthesis.
- `--ollama-url`: base URL of the Ollama instance (default `http://localhost:11434`). Point this at your remote host when needed, e.g. `http://10.0.0.5:11434`.
- `--ollama-max-tokens`: cap for the Ollama response (`num_predict`, default `400`).

If you omit `--ollama-model`, the tool prints the ranked chunks plus a quick summary of where to look in the codebase. This still gives you a useful retrieval workflow entirely offline.

### 3. Inspect stored rows

```bash
python list_chroma_rows.py --limit 10 --show-docs
```

Flags:

- `--limit` / `--offset`: paginate through the stored rows (defaults 20 / 0).
- `--show-docs`: include the actual chunk text in addition to metadata.
- `--json`: emit a JSON blob that you can pipe into `jq` or other tooling.

## Example Flow

```bash
# 1. Index the repo (only needs to happen when files change)
python chromadb_code_rag.py . --chunk-lines 60 --chunk-overlap 15 --reset

# 2. Ask questions locally
python query_tool.py "Where are database connections created?" --top-k 3

# 3. (Optional) Let a remote Ollama host draft an answer
python query_tool.py --ollama-url localhost:11434 --ollama-model gpt-oss:20b-cloud "Which are the cities that are in this simulation?"
```

## How It Works

1. **Chunking** - Files are split into overlapping windows of configurable line counts so embeddings capture enough context.
2. **Embedding** - Uses ChromaDB's `DefaultEmbeddingFunction`, which bundles a light-weight SentenceTransformer and requires no external services.
3. **Storage** - Chunks live in a persistent Chroma collection, so you only ingest when the code changes.
4. **Retrieval** - Queries are embedded with the same model; the nearest neighbors provide the best-matching code fragments plus file/line references.
5. **Synthesis (optional)** - Send the retrieved context to your Ollama server for a full natural-language answer, or review the snippets yourself.

Because each script is a single, well-scoped Python file, you can tweak them easily - for example swapping in another embedding model, filtering additional file types, or wiring them into a larger automation or CI pipeline.

## MCP Server (Experimental)

This repo also ships an [MCP](https://modelcontextprotocol.io/docs/develop/build-server) server (`chromadb_mcp_server.py`) built with `FastMCP`. It exposes the existing Chroma workflows as first-class MCP tools so desktop clients such as Claude can request context directly from your local database.

Available tools:

- `list_collections` — enumerates persisted Chroma collections and their row counts.
- `query_codebase` — embeds a natural-language question and returns the top matching chunks plus a quick summary.
- `list_rows` — inspects raw rows/documents from a collection (useful for debugging what was ingested).

### Running the server

1. Install dependencies (`pip install -r requirements.txt`).
2. Make sure you have already ingested the target repository with `chromadb_code_rag.py`.
3. Start the server:  
   ```bash
   python chromadb_mcp_server.py
   ```
4. Point your MCP-compatible client at the script. For example, in `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "code-rag": {
         "command": "python",
         "args": [
           "C:/absolute/path/to/mcp_example/chromadb_mcp_server.py"
         ]
       }
     }
   }
   ```

Claude (or another MCP client) will now be able to call `list_collections`, `query_codebase`, and `list_rows` over STDIO, retrieving the same snippets that `query_tool.py` prints to the console.
