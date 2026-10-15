"""Tests for skill discovery and the model harness."""

from __future__ import annotations

import pytest

from backend.ollama_client import OllamaClient, same_origin
from backend.orchestrator import (
    AgentOrchestrator,
    SkillNotFoundError,
    SkillValidationError,
)

EXPECTED_SKILLS = {
    "text_summarizer",
    "code_reviewer",
    "github_pr_summarizer",
    "rag_doc_search",
}


@pytest.fixture()
def harness() -> AgentOrchestrator:
    """Build a harness pointed at an unused local port so Ollama is offline."""
    return AgentOrchestrator("http://127.0.0.1:9")


def test_discovery_finds_four_skills(harness: AgentOrchestrator) -> None:
    """The loader registers each bundled skill by its standard name."""
    assert set(harness.skills) == EXPECTED_SKILLS
    names = {entry["name"] for entry in harness.list_skills()}
    assert names == EXPECTED_SKILLS
    assert all(entry["standard"].startswith("Agent Skill Open Standard") for entry in harness.list_skills())


def test_execute_skill_success_and_failure(harness: AgentOrchestrator) -> None:
    """A valid summary succeeds, and unknown or invalid calls raise."""
    output = harness.execute_skill(
        "text_summarizer",
        {"text": "Agents observe. Agents choose a tool. Agents check the result."},
        False,
        None,
        "llama3",
        0.7,
    )
    assert output["skill_name"] == "text_summarizer"
    assert isinstance(output["result"], str)
    assert output["result"]
    assert output["llm_response"] is None
    assert output["metrics"]["skill_version"] == "1.0.0"
    assert output["metrics"]["estimated_tokens"] >= 1
    assert harness.metrics.total_requests == 1

    with pytest.raises(SkillNotFoundError):
        harness.execute_skill("missing_skill", {}, False, None, "llama3", 0.7)
    with pytest.raises(SkillValidationError):
        harness.execute_skill("text_summarizer", {}, False, None, "llama3", 0.7)


def test_execute_skill_with_offline_model(harness: AgentOrchestrator) -> None:
    """LLM enhancement falls back cleanly when Ollama is not running."""
    output = harness.execute_skill(
        "text_summarizer",
        {"text": "Please summarize this short note about local agents and tools."},
        True,
        None,
        "llama3",
        0.2,
    )
    assert output["llm_source"] == "fallback"
    assert output["llm_response"]
    assert output["metrics"]["model_used"] == "llama3-mock"


def test_chat_maps_retrieval_query(harness: AgentOrchestrator) -> None:
    """Chat with the retrieval skill uses the message as the query."""
    output = harness.chat(
        "How does chunking reduce hallucination during retrieval?",
        "rag_doc_search",
        "llama3",
        0.2,
    )
    assert output["skill_used"] == "rag_doc_search"
    assert "rag_explained.txt" in output["response"]
    assert "LLM insight" in output["response"]


def test_get_status_shape(harness: AgentOrchestrator) -> None:
    """Status reports connectivity, skills, counters, and uptime."""
    status = harness.get_status()
    assert set(status) >= {
        "ollama_connected",
        "ollama_models",
        "loaded_skills",
        "total_requests",
        "uptime_seconds",
    }
    assert status["ollama_connected"] is False
    assert status["ollama_models"] == []
    assert set(status["loaded_skills"]) == EXPECTED_SKILLS
    assert status["total_requests"] == 0
    assert status["uptime_seconds"] >= 0


def test_same_origin_detects_the_api_host() -> None:
    """The dashboard must not treat the FastAPI URL as an Ollama server."""
    api = "https://agentflow-api-m9lc.onrender.com"
    assert same_origin(api, api + "/")
    assert same_origin(api + "/api/tags", api)
    assert not same_origin("http://127.0.0.1:11434", api)
    assert not same_origin("", api)


def test_fallback_detects_summary_prompt() -> None:
    """The offline client labels a summary-style prompt and stays local."""
    client = OllamaClient("http://127.0.0.1:9")
    healthy, models = client.health_check()
    assert healthy is False
    assert models == []
    generated = client.smart_generate("Please summarize the agent design for the demo.")
    assert generated["source"] == "fallback"
    assert generated["model"] == "llama3-mock"
    assert "Summary" in generated["response"]
