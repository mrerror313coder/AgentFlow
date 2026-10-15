"""Local TF-IDF retrieval skill.

Indexes ``.txt`` and ``.md`` files from a directory, chunks them, and ranks
chunks against a query with cosine similarity. Term frequency, inverse
document frequency, and cosine similarity are computed with the standard
library only.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from skills.base_skill import BaseSkill

logger = logging.getLogger("agentflow.skills.rag_doc_search")

_TOKEN = re.compile(r"[a-z0-9]+")
_CHUNK_WORDS = 200
_CHUNK_OVERLAP = 50
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "these",
        "those",
        "not",
    }
)


class RAGDocSearch(BaseSkill):
    """Retrieve the most similar local document chunks for a query."""

    def __init__(self) -> None:
        """Create a skill with an empty in-memory index cache."""
        self._index: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        """Registry id for local document search."""
        return "rag_doc_search"

    @property
    def description(self) -> str:
        """Short description shown in the skill registry."""
        return (
            "Search local notes with a from-scratch TF-IDF index and return "
            "the most similar text chunks."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for retrieval input."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "data_dir": {
                    "type": "string",
                    "description": "Directory of .txt and .md notes. Defaults to data/sample_notes.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return. Defaults to 3.",
                },
            },
            "required": ["query"],
        }

    @property
    def version(self) -> str:
        """Semantic version of the retrieval skill."""
        return "1.0.0"

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Index ``data_dir`` if needed and return the top matching chunks.

        Args:
            input_data: Must include ``query``. ``data_dir`` and ``top_k``
                are optional.

        Returns:
            Ranked matches and the number of chunks in the index.

        Raises:
            ValueError: If the input does not match ``input_schema``.
        """
        self.ensure_valid(input_data)
        started = time.perf_counter()
        query = str(input_data["query"]).strip()
        data_dir = str(input_data.get("data_dir") or "data/sample_notes")
        top_k = max(1, int(input_data.get("top_k", 3)))
        resolved = self._resolve_data_dir(data_dir)

        index, build_ms, cached = self._get_index(resolved)
        matches = self._search(index, query, top_k)
        return {
            "result": {
                "query": query,
                "matches": matches,
                "total_chunks": len(index["chunks"]),
            },
            "metadata": self.build_metadata(
                started,
                index_build_time_ms=build_ms,
                data_dir=str(resolved),
                cached=cached,
                top_k=top_k,
            ),
        }

    def _get_index(self, data_dir: Path) -> tuple[dict[str, Any], float, bool]:
        """Return the cached index or build a new one.

        Args:
            data_dir: Absolute directory that should be indexed.

        Returns:
            The index, milliseconds spent building it, and whether the cache hit.
        """
        key = str(data_dir)
        if self._index is not None and self._index.get("data_dir") == key:
            return self._index, 0.0, True
        started = time.perf_counter()
        index = self._build_index(data_dir)
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        self._index = index
        logger.info(
            "Built TF-IDF index for %s with %d chunk(s) in %sms",
            data_dir,
            len(index["chunks"]),
            elapsed,
        )
        return index, elapsed, False

    def _build_index(self, data_dir: Path) -> dict[str, Any]:
        """Read notes, chunk them, and store a TF-IDF vector per chunk."""
        chunks: list[dict[str, Any]] = []
        if data_dir.is_dir():
            files = sorted(
                path
                for path in data_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".txt", ".md"}
            )
        else:
            logger.warning("RAG data directory does not exist: %s", data_dir)
            files = []

        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read %s: %s", path, exc)
                continue
            relative = str(path.relative_to(data_dir))
            for chunk_index, chunk_text in enumerate(self._chunk(text)):
                tokens = self._tokenize(chunk_text)
                counts = Counter(tokens)
                total = sum(counts.values()) or 1
                tf = {term: count / total for term, count in counts.items()}
                chunks.append(
                    {
                        "source": relative,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "tf": tf,
                        "terms": set(tf),
                    }
                )

        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            document_frequency.update(chunk["terms"])
        total_chunks = len(chunks)
        idf = {
            term: math.log(total_chunks / (1 + doc_count))
            for term, doc_count in document_frequency.items()
        }
        for chunk in chunks:
            chunk["vector"] = {
                term: weight * idf.get(term, 0.0) for term, weight in chunk["tf"].items()
            }
            chunk.pop("tf", None)
            chunk.pop("terms", None)
        return {"data_dir": str(data_dir), "chunks": chunks, "idf": idf}

    def _search(self, index: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
        """Rank chunks by cosine similarity to the query vector."""
        counts = Counter(self._tokenize(query))
        total = sum(counts.values()) or 1
        idf: dict[str, float] = index["idf"]
        query_vector = {
            term: (count / total) * idf[term]
            for term, count in counts.items()
            if term in idf
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in index["chunks"]:
            score = self._cosine(query_vector, chunk["vector"])
            scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["chunk_index"]))
        matches: list[dict[str, Any]] = []
        for score, chunk in scored[:top_k]:
            matches.append(
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "score": round(score, 6),
                    "text": chunk["text"],
                }
            )
        return matches

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        """Cosine similarity of two sparse term vectors."""
        if not left or not right:
            return 0.0
        dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
        left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
        right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric tokens with stopwords removed."""
        return [
            token
            for token in _TOKEN.findall(text.lower())
            if token not in STOPWORDS and len(token) > 1
        ]

    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Split text into overlapping word windows of about 200 words."""
        words = text.split()
        if not words:
            return []
        if len(words) <= _CHUNK_WORDS:
            return [" ".join(words)]
        chunks: list[str] = []
        start = 0
        step = _CHUNK_WORDS - _CHUNK_OVERLAP
        while start < len(words):
            end = min(start + _CHUNK_WORDS, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += step
        return chunks

    @staticmethod
    def _resolve_data_dir(data_dir: str) -> Path:
        """Resolve a data directory from the cwd or the project root."""
        candidate = Path(data_dir)
        if candidate.is_dir():
            return candidate.resolve()
        rooted = _PROJECT_ROOT / data_dir
        if rooted.is_dir():
            return rooted.resolve()
        return candidate if candidate.is_absolute() else rooted
