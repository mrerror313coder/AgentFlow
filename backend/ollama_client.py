"""Ollama client with a prompt-aware offline fallback.

Live inference uses the local Ollama HTTP API. Any connection error, HTTP
error, invalid payload, or empty completion is converted into a labeled
fallback response so the dashboard and tests still run without a model.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger("agentflow.ollama")


def same_origin(left: str, right: str) -> bool:
    """Return whether two URLs point at the same host and port.

    Args:
        left: First URL or origin.
        right: Second URL or origin.

    Returns:
        True when both have a network location and those locations match.
    """
    left_host = urlparse((left or "").strip()).netloc.lower()
    right_host = urlparse((right or "").strip()).netloc.lower()
    return bool(left_host) and left_host == right_host

_HEALTH_TIMEOUT_SECONDS = 3
_GENERATE_TIMEOUT_SECONDS = 120

_SUMMARY_KEYS = ("summarize", "summary")
_REVIEW_KEYS = ("review",)
_CODE_KEYS = ("code", "bug", "error")
_COMPARE_KEYS = ("compare",)
_EXPLAIN_KEYS = ("explain", "what is", "how to")
_LIST_KEYS = ("list",)
_HELP_KEYS = ("help",)

_STOPWORDS = frozenset(
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
        "this",
        "that",
        "these",
        "those",
        "it",
        "please",
        "what",
        "how",
        "does",
        "when",
        "where",
        "which",
        "into",
        "following",
        "analyze",
        "provide",
        "insights",
        "about",
        "your",
        "query",
    }
)


class OllamaClient:
    """Talk to a local Ollama server, or answer offline when it cannot."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        """Store the Ollama origin, without a trailing slash.

        Args:
            base_url: Ollama server origin, for example ``http://localhost:11434``.
        """
        self.base_url = base_url.rstrip("/")
        self._health_cache: tuple[float, tuple[bool, list[str]]] | None = None

    def health_check(self) -> tuple[bool, list[str]]:
        """Check ``GET /api/tags`` and return the installed model names.

        A successful or failed result is reused for five seconds so the
        dashboard can render without waiting on Ollama for every widget.

        Returns:
            ``(True, model_names)`` when Ollama responds, otherwise ``(False, [])``.
        """
        now = time.monotonic()
        if self._health_cache is not None and now - self._health_cache[0] < 5:
            return self._health_cache[1]
        result = self._health_check_live()
        self._health_cache = (now, result)
        return result

    def _health_check_live(self) -> tuple[bool, list[str]]:
        """Call Ollama's tag endpoint once."""
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=_HEALTH_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.warning("Ollama health check failed for %s: %s", url, exc)
            return False, []
        except ValueError as exc:
            logger.warning("Ollama health check returned invalid JSON: %s", exc)
            return False, []

        models: list[str] = []
        for item in payload.get("models", []) if isinstance(payload, dict) else []:
            if isinstance(item, dict) and item.get("name"):
                models.append(str(item["name"]))
        logger.info("Ollama is reachable at %s with %d model(s)", self.base_url, len(models))
        return True, models

    def generate(
        self,
        prompt: str,
        model: str = "llama3",
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate a completion from Ollama.

        Args:
            prompt: Full prompt sent to the model.
            model: Ollama model name.
            temperature: Sampling temperature forwarded in ``options``.

        Returns:
            A dictionary with ``response``, ``model``, ``source``, and ``eval_count``.

        Raises:
            RuntimeError: The server could not be reached or returned an error.
        """
        url = f"{self.base_url}/api/generate"
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(url, json=body, timeout=_GENERATE_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Ollama generate returned invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama generate returned an unexpected payload.")
        text = str(payload.get("response") or "")
        logger.info("Ollama generate completed model=%s eval_count=%s", model, payload.get("eval_count"))
        return {
            "response": text,
            "model": str(payload.get("model") or model),
            "source": "ollama",
            "eval_count": int(payload.get("eval_count") or 0),
        }

    def fallback_generate(self, prompt: str, model: str = "llama3") -> dict[str, Any]:
        """Build a deterministic answer from keywords in the prompt.

        Args:
            prompt: The same prompt that would have been sent to Ollama.
            model: Requested model name, recorded with a ``-mock`` suffix.

        Returns:
            A labeled fallback completion. ``source`` is always ``fallback``.
        """
        category = self._category(prompt)
        topic = self._topic(prompt)
        highlights = self._highlights(prompt)
        response = self._template(category, topic, highlights)
        logger.info("Using offline fallback category=%s model=%s", category, model)
        return {
            "response": response,
            "model": f"{model}-mock",
            "source": "fallback",
            "eval_count": 0,
        }

    def smart_generate(
        self,
        prompt: str,
        model: str = "llama3",
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Prefer Ollama, and fall back when live inference is unavailable.

        An empty completion is treated as a failure. The returned ``source``
        field tells the dashboard which path produced the text.

        Args:
            prompt: Prompt to complete.
            model: Ollama model name.
            temperature: Sampling temperature for the live call.

        Returns:
            Either an Ollama completion or a fallback completion.
        """
        logger.info("Attempting Ollama generate model=%s", model)
        healthy, _models = self.health_check()
        if not healthy:
            logger.warning("Ollama is unreachable at %s; using fallback", self.base_url)
            return self.fallback_generate(prompt, model=model)
        try:
            result = self.generate(prompt, model=model, temperature=temperature)
        except Exception as exc:
            logger.warning("Ollama generate failed (%s); using fallback", exc)
            return self.fallback_generate(prompt, model=model)
        if not str(result.get("response") or "").strip():
            logger.warning("Ollama returned an empty completion; using fallback")
            return self.fallback_generate(prompt, model=model)
        return result

    @staticmethod
    def _category(prompt: str) -> str:
        """Pick a response template from keywords in the prompt."""
        lowered = prompt.lower()
        checks = (
            ("summary", _SUMMARY_KEYS),
            ("review", _REVIEW_KEYS),
            ("code", _CODE_KEYS),
            ("compare", _COMPARE_KEYS),
            ("explain", _EXPLAIN_KEYS),
            ("list", _LIST_KEYS),
            ("help", _HELP_KEYS),
        )
        for name, keys in checks:
            if any(key in lowered for key in keys):
                return name
        return "general"

    @staticmethod
    def _topic(prompt: str) -> str:
        """Extract a short topic phrase for the fallback template."""
        cleaned = re.sub(r"\s+", " ", prompt).strip()
        quoted = re.search(r"[\"']([^\"']{3,80})[\"']", cleaned)
        if quoted:
            return quoted.group(1).strip()
        sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0]
        return sentence[:140].strip() or "the request"

    @staticmethod
    def _highlights(prompt: str) -> list[str]:
        """Collect a few distinctive words so the fallback reflects the prompt."""
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", prompt)
        seen: list[str] = []
        for word in words:
            if word.lower() in _STOPWORDS:
                continue
            if word.lower() in {item.lower() for item in seen}:
                continue
            seen.append(word)
            if len(seen) == 4:
                break
        return seen or ["the main idea", "the supporting detail", "the practical takeaway"]

    @staticmethod
    def _template(category: str, topic: str, highlights: list[str]) -> str:
        """Render a category template using the extracted topic and highlights."""
        first, second, third = (highlights + ["context", "detail", "next step"])[:3]
        if category == "summary":
            return (
                f"**Summary:** The provided content covers several key aspects of {topic}. "
                f"[1] The primary focus is on {first}. "
                f"[2] Key details include {second}. "
                f"[3] The conclusion suggests paying attention to {third}."
            )
        if category == "review":
            return (
                f"**Review:** Looking at {topic}, the change is understandable but worth a second pass. "
                f"[1] Confirm the behavior around {first}. "
                f"[2] Check edge cases related to {second}. "
                f"[3] Note how {third} affects readability and risk."
            )
        if category == "code":
            return (
                "**Code Analysis:** The code appears to implement a focused piece of behavior. "
                "Potential improvements: "
                f"[1] Consider adding error handling around {first}. "
                f"[2] Variable naming could be more descriptive where {second} is used. "
                f"[3] A small test covering {third} would lock the expected result in place."
            )
        if category == "compare":
            return (
                f"**Comparison:** For {topic}, the useful differences sit in scope and tradeoffs. "
                f"[1] One side emphasizes {first}. "
                f"[2] The other side emphasizes {second}. "
                f"[3] Choose based on how much {third} matters in your workflow."
            )
        if category == "explain":
            return (
                f"**Explanation:** {topic} can be understood in three steps. "
                f"[1] Start with {first}, which is the core idea. "
                f"[2] Then look at {second}, which is how the pieces connect. "
                f"[3] Finally, {third} is what you would change or verify in practice."
            )
        if category == "list":
            return (
                f"**List:** Based on {topic}, here are the items that stand out. "
                f"[1] {first}. "
                f"[2] {second}. "
                f"[3] {third}."
            )
        if category == "help":
            return (
                f"**Help:** Here is a practical way to approach {topic}. "
                f"[1] Identify where {first} shows up. "
                f"[2] Try the smallest change that exercises {second}. "
                f"[3] Check the result against {third} before expanding the scope."
            )
        return (
            f"Based on your query about '{topic}', here are the key points: "
            f"[1] {first} is the part to anchor on. "
            f"[2] {second} adds the context you need. "
            f"[3] {third} is the next concrete step."
        )
