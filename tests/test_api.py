"""HTTP tests for the AgentFlow FastAPI harness."""

from __future__ import annotations

import json
import os

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture()
def client() -> TestClient:
    """Open a test client so the startup hook builds a harness."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_and_health(client: TestClient) -> None:
    """The welcome route and health route both respond with the live registry."""
    root = client.get("/")
    assert root.status_code == 200
    body = root.json()
    assert body["version"] == "2.0.0"
    assert "Agent Skill Open Standard" in body["standard"]

    health = client.get("/api/health")
    assert health.status_code == 200
    status = health.json()
    assert len(status["loaded_skills"]) == 4
    assert "total_requests" in status
    assert "uptime_seconds" in status


def test_skill_registry(client: TestClient) -> None:
    """The registry lists the four bundled skills and their schemas."""
    response = client.get("/api/skills")
    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert names == {
        "text_summarizer",
        "code_reviewer",
        "github_pr_summarizer",
        "rag_doc_search",
    }
    assert all(item["standard"] == "Agent Skill Open Standard v1.0" for item in response.json())


def test_execute_success_not_found_and_bad_input(client: TestClient) -> None:
    """Execution returns 200, 404, and 400 for the three main outcomes."""
    ok = client.post(
        "/api/skills/execute",
        json={
            "skill_name": "text_summarizer",
            "input_data": {"text": "Agents plan. Agents act. Agents observe the tool result."},
            "use_llm": False,
        },
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["skill_name"] == "text_summarizer"
    assert payload["result"]
    assert payload["metrics"]["estimated_tokens"] >= 1

    missing = client.post(
        "/api/skills/execute",
        json={"skill_name": "does_not_exist", "input_data": {}},
    )
    assert missing.status_code == 404

    invalid = client.post(
        "/api/skills/execute",
        json={"skill_name": "text_summarizer", "input_data": {}},
    )
    assert invalid.status_code == 400


def test_chat_direct_and_with_skill(client: TestClient) -> None:
    """Direct chat and skill chat both return text when Ollama is offline."""
    direct = client.post("/api/chat", json={"message": "What is a language model?"})
    assert direct.status_code == 200
    assert direct.json()["response"]
    assert direct.json()["skill_used"] is None

    skilled = client.post(
        "/api/chat",
        json={
            "message": "How does chunking reduce hallucination during retrieval?",
            "skill_name": "rag_doc_search",
        },
    )
    assert skilled.status_code == 200
    assert skilled.json()["skill_used"] == "rag_doc_search"
    assert "LLM insight" in skilled.json()["response"]

    unknown = client.post(
        "/api/chat",
        json={"message": "hello", "skill_name": "missing_skill"},
    )
    assert unknown.status_code == 404


def test_metrics_history_and_export(client: TestClient) -> None:
    """History grows after a call, and the export endpoint downloads JSON."""
    client.post(
        "/api/skills/execute",
        json={
            "skill_name": "code_reviewer",
            "input_data": {"code": "def ok():\n    \"\"\"Fine.\"\"\"\n    return 1\n"},
            "use_llm": False,
        },
    )
    history = client.get("/api/metrics")
    assert history.status_code == 200
    body = history.json()
    assert body["total_requests"] >= 1
    assert body["history"][-1]["skill_name"] == "code_reviewer"

    exported = client.get("/api/metrics/export")
    assert exported.status_code == 200
    assert "application/json" in exported.headers["content-type"]
    assert "attachment" in exported.headers["content-disposition"]
    parsed = json.loads(exported.content)
    assert parsed["total_requests"] >= 1
    assert parsed["history"]
