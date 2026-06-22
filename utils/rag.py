"""
Lightweight, in-memory RAG pipeline for uploaded PDF(s).

No vector database on purpose: documents are session-only (cleared when the
app restarts), so a plain numpy cosine-similarity search is simpler, has zero
extra moving parts, and is fast enough for the size of documents a chat UI
typically sees.

Two retrieval modes:
  - normal Q&A -> similarity search picks the most relevant chunks
  - summarization requests ("summarize this in 3 lines") -> similarity
    search would just grab a near-random subset of chunks, since there's no
    real semantic target to match against. Instead, feed the whole document
    in directly if it fits, or map-reduce summarize it first if it doesn't.

Requires an embedding model pulled in Ollama, e.g.:
    ollama pull nomic-embed-text
"""

import io

import numpy as np
from pypdf import PdfReader

from utils.ollama_client import embed_text

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Rough heuristic: small/fast local models have limited context windows.
# Stay well under it so there's still room left for the system prompt,
# chat history, and the response itself.
MAX_DIRECT_SUMMARY_CHARS = 12000

SUMMARY_KEYWORDS = [
    "summarize", "summarise", "summary", "overview", "main points",
    "key points", "tl;dr", "tldr", "in short", "gist", "brief",
]


def is_summarization_request(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in SUMMARY_KEYWORDS)


def extract_pdf_text(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples for a PDF's contents."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def full_text_from_pages(pages: list[tuple[int, str]]) -> str:
    """Plain concatenation of page text — used for summarization, NOT for
    chunk-based retrieval (chunks overlap on purpose; this shouldn't)."""
    return "\n\n".join(text for _, text in pages)


def chunk_text(pages: list[tuple[int, str]]) -> list[dict]:
    """Split page text into overlapping chunks, keeping track of the source page."""
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"page": page_num, "text": chunk})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index(chunks: list[dict]) -> np.ndarray:
    """Embed every chunk and return a (n_chunks, dim) numpy array."""
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = [embed_text(EMBED_MODEL, c["text"]) for c in chunks]
    return np.array(vectors, dtype=np.float32)


def retrieve(query: str, chunks: list[dict], embeddings: np.ndarray, top_k: int = 4) -> list[dict]:
    """Return the top_k most relevant chunks for a query via cosine similarity.
    Use this for normal Q&A — NOT for summarization requests."""
    if embeddings is None or len(chunks) == 0 or embeddings.size == 0:
        return []

    query_vec = np.array(embed_text(EMBED_MODEL, query), dtype=np.float32)

    chunk_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)
    denom = chunk_norms * query_norm
    denom[denom == 0] = 1e-8

    scores = (embeddings @ query_vec) / denom
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def summarize_document(model: str, full_text: str, chat_fn) -> str:
    """
    Returns text suitable for injecting into the main prompt as context for
    a summarization request.

    - If the document is short enough, just return the full text directly —
      let the main chat call do the actual summarizing, with full context.
    - If it's too long, map-reduce: summarize each piece first (using
      `chat_fn`, a non-streaming call so this doesn't show up as a separate
      visible reply), then return the combined partial summaries as context.

    `chat_fn` should have the same signature as ollama_client.chat_once.
    """
    if len(full_text) <= MAX_DIRECT_SUMMARY_CHARS:
        return full_text

    pieces = [
        full_text[i:i + MAX_DIRECT_SUMMARY_CHARS]
        for i in range(0, len(full_text), MAX_DIRECT_SUMMARY_CHARS)
    ]
    partial_summaries = []
    for piece in pieces:
        partial = chat_fn(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "Summarize the following document excerpt in 4-6 bullet "
                        "points, capturing only the key facts. Be concise."
                    ),
                },
                {"role": "user", "content": piece},
            ],
        )
        partial_summaries.append(partial)

    return "\n\n".join(partial_summaries)


def process_uploaded_pdf(file_bytes: bytes) -> tuple[list[dict], np.ndarray, str]:
    """Full pipeline: PDF bytes -> page text -> chunks -> embeddings -> full text."""
    pages = extract_pdf_text(file_bytes)
    chunks = chunk_text(pages)
    embeddings = build_index(chunks)
    full_text = full_text_from_pages(pages)
    return chunks, embeddings, full_text
