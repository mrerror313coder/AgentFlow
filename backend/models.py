"""Pydantic request and response models for the AgentFlow API.

The shapes here are the public contract between the Streamlit dashboard and
the FastAPI harness. Skill results stay flexible because individual skills
return either text or structured objects.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExecutionMetrics(BaseModel):
    """Timing and token estimate recorded for one harness call."""

    execution_time_ms: float
    estimated_tokens: int
    model_used: str | None = None
    skill_version: str
    timestamp: str


class SkillExecutionRequest(BaseModel):
    """Request body for ``POST /api/skills/execute``."""

    skill_name: str
    input_data: dict[str, Any]
    use_llm: bool = False
    llm_prompt_template: str | None = None
    model: str = "llama3"
    temperature: float = 0.7


class SkillExecutionResponse(BaseModel):
    """Response body for a single skill execution."""

    skill_name: str
    result: Any
    metrics: ExecutionMetrics
    llm_response: str | None = None
    llm_source: str | None = None


class ChatRequest(BaseModel):
    """Request body for ``POST /api/chat``."""

    message: str
    skill_name: str | None = None
    model: str = "llama3"
    temperature: float = 0.7


class ChatResponse(BaseModel):
    """Response body for a chat turn."""

    response: str
    skill_used: str | None = None
    metrics: ExecutionMetrics


class SkillRegistryEntry(BaseModel):
    """One discoverable skill advertised by the registry."""

    name: str
    description: str
    input_schema: dict[str, Any]
    version: str
    standard: str = "Agent Skill Open Standard v1.0"


class SystemStatus(BaseModel):
    """Process health snapshot returned by ``GET /api/health``."""

    ollama_connected: bool
    ollama_models: list[str] = Field(default_factory=list)
    loaded_skills: list[str] = Field(default_factory=list)
    total_requests: int
    uptime_seconds: float
