"""Tests for skill validation, execution, retrieval, and mock pull requests."""

from __future__ import annotations

from pathlib import Path

from skills.code_reviewer import CodeReviewer
from skills.github_pr_summarizer import GitHubPRSummarizer
from skills.rag_doc_search import RAGDocSearch
from skills.summarizer import TextSummarizer

NOTES = Path(__file__).resolve().parents[1] / "data" / "sample_notes"

REVIEW_SAMPLE = '''from os import *

def bad(items=[]):
    try:
        return items[0]
    except:
        return None

def long_one():
    """Documented helper used to cross the length threshold."""
    first = 1
    second = 2
    third = 3
    fourth = 4
    fifth = 5
    sixth = 6
    seventh = 7
    eighth = 8
    ninth = 9
    tenth = 10
    eleventh = 11
    twelfth = 12
    thirteenth = 13
    fourteenth = 14
    fifteenth = 15
    sixteenth = 16
    seventeenth = 17
    eighteenth = 18
    nineteenth = 19
    twentieth = 20
    return first + twentieth
'''


def test_summarizer_validation() -> None:
    """Accept a string payload and reject a missing or mistyped field."""
    skill = TextSummarizer()
    ok, message = skill.validate_input({"text": "Agents plan and act."})
    assert ok is True
    assert message == ""
    ok, message = skill.validate_input({})
    assert ok is False
    assert "text" in message
    ok, message = skill.validate_input({"text": "Hello.", "max_sentences": True})
    assert ok is False
    assert "max_sentences" in message


def test_summarizer_execute_keeps_original_order() -> None:
    """High-scoring banana sentences stay in the order they were written."""
    skill = TextSummarizer()
    text = (
        "Bananas are yellow and sweet fruit. "
        "The zebra is unrelated filler here today. "
        "Bananas grow in warm climates worldwide."
    )
    output = skill.execute({"text": text, "max_sentences": 2})
    summary = output["result"]
    assert "Bananas are yellow" in summary
    assert "Bananas grow" in summary
    assert "zebra" not in summary
    assert summary.index("yellow") < summary.index("grow")
    meta = output["metadata"]
    assert meta["total_sentences"] == 3
    assert meta["selected_sentences"] == 2
    assert meta["skill_version"] == "1.0.0"
    assert meta["execution_time_ms"] >= 0


def test_code_reviewer_validation_and_findings() -> None:
    """Reject a missing code field and flag the sample's known rules."""
    skill = CodeReviewer()
    ok, _message = skill.validate_input({"code": "x = 1\n"})
    assert ok is True
    ok, message = skill.validate_input({"code": 12})
    assert ok is False
    assert "code" in message
    output = skill.execute({"code": REVIEW_SAMPLE, "language": "python"})
    rules = {item["rule"] for item in output["result"]["findings"]}
    assert "star_import" in rules
    assert "mutable_default" in rules
    assert "bare_except" in rules
    assert "missing_docstring" in rules
    assert "long_function" in rules
    assert "summary" in output["result"]
    assert output["metadata"]["skill_version"] == skill.version


def test_pull_request_mock_mode() -> None:
    """Mock mode parses unified diffs into additions, deletions, and a narrative."""
    skill = GitHubPRSummarizer()
    ok, message = skill.validate_input({"repo": "octocat/Hello-World"})
    assert ok is False
    assert "pr_number" in message
    ok, _message = skill.validate_input({"repo": "octocat/Hello-World", "pr_number": 1})
    assert ok is True
    output = skill.execute({"repo": "octocat/Hello-World", "pr_number": 1})
    result = output["result"]
    assert output["metadata"]["source"] == "mock"
    assert 3 <= result["total_files"] <= 5
    assert result["total_additions"] == sum(item["additions"] for item in result["files"])
    assert result["total_deletions"] == sum(item["deletions"] for item in result["files"])
    assert result["total_additions"] > 0
    assert "This PR modifies" in result["narrative"]
    assert any(item["filename"] == "src/auth/login.py" for item in result["files"])


def test_rag_retrieves_relevant_notes() -> None:
    """TF-IDF ranks the transformer note and the retrieval note for matching queries."""
    skill = RAGDocSearch()
    ok, message = skill.validate_input({})
    assert ok is False
    assert "query" in message
    llm = skill.execute(
        {
            "query": "How do transformers and attention work in Llama and Mistral?",
            "data_dir": str(NOTES),
            "top_k": 2,
        }
    )
    assert llm["metadata"]["index_build_time_ms"] >= 0
    assert llm["result"]["total_chunks"] >= 3
    assert llm["result"]["matches"]
    assert llm["result"]["matches"][0]["source"].endswith("llm_introduction.txt")
    assert llm["result"]["matches"][0]["score"] > 0

    rag = skill.execute(
        {
            "query": "How does chunking reduce hallucination during retrieval?",
            "data_dir": str(NOTES),
            "top_k": 1,
        }
    )
    assert rag["metadata"]["cached"] is True
    assert rag["result"]["matches"][0]["source"].endswith("rag_explained.txt")


def test_registry_entries_name_the_standard() -> None:
    """Every bundled skill advertises the open standard in its registry entry."""
    skills = [TextSummarizer(), CodeReviewer(), GitHubPRSummarizer(), RAGDocSearch()]
    for skill in skills:
        entry = skill.to_registry_entry()
        assert entry["name"] == skill.name
        assert entry["standard"] == "Agent Skill Open Standard v1.0"
        assert entry["input_schema"]["type"] == "object"
        assert entry["version"] == "1.0.0"
