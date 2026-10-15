"""Extractive text summarizer skill.

Scores sentences by word frequency and returns the highest-scoring sentences
in their original order. No model and no third-party NLP library is required.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from skills.base_skill import BaseSkill

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
        "so",
        "than",
        "can",
        "will",
    }
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9']+")


class TextSummarizer(BaseSkill):
    """Select the most informative sentences from a passage."""

    @property
    def name(self) -> str:
        """Registry id for the summarizer."""
        return "text_summarizer"

    @property
    def description(self) -> str:
        """Short description shown in the skill registry."""
        return (
            "Extractive summarizer that ranks sentences by word frequency "
            "and keeps the top sentences in their original order."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for summarizer input."""
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Source text to summarize.",
                },
                "max_sentences": {
                    "type": "integer",
                    "description": "Maximum number of sentences to keep. Defaults to 3.",
                },
            },
            "required": ["text"],
        }

    @property
    def version(self) -> str:
        """Semantic version of the summarizer."""
        return "1.0.0"

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize ``text`` down to ``max_sentences`` sentences.

        Args:
            input_data: Must include ``text``. ``max_sentences`` is optional.

        Returns:
            The joined summary plus counts of total and selected sentences.

        Raises:
            ValueError: If the input does not match ``input_schema``.
        """
        self.ensure_valid(input_data)
        started = time.perf_counter()
        text = str(input_data["text"]).strip()
        max_sentences = int(input_data.get("max_sentences", 3))
        max_sentences = max(1, max_sentences)

        sentences = [
            piece.strip()
            for piece in _SENTENCE_SPLIT.split(text)
            if piece.strip()
        ]
        if not sentences:
            return {
                "result": "",
                "metadata": self.build_metadata(
                    started,
                    total_sentences=0,
                    selected_sentences=0,
                ),
            }

        tokenized = [self._tokens(sentence) for sentence in sentences]
        frequencies: Counter[str] = Counter()
        for tokens in tokenized:
            frequencies.update(tokens)

        scores: list[float] = []
        for tokens in tokenized:
            if not tokens:
                scores.append(0.0)
                continue
            scores.append(sum(frequencies[token] for token in tokens) / len(tokens))

        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        chosen_indexes = sorted(index for index, _score in ranked[:max_sentences])
        selected = [sentences[index] for index in chosen_indexes]
        summary = " ".join(selected)
        return {
            "result": summary,
            "metadata": self.build_metadata(
                started,
                total_sentences=len(sentences),
                selected_sentences=len(selected),
            ),
        }

    @staticmethod
    def _tokens(sentence: str) -> list[str]:
        """Lowercase a sentence and drop stopwords and one-character tokens."""
        return [
            token
            for token in _WORD.findall(sentence.lower())
            if token not in STOPWORDS and len(token) > 1
        ]
