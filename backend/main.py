"""FastAPI server for the AgentFlow model harness.

The process creates one orchestrator at startup. Skill errors become HTTP 404
and 400 responses. Unexpected failures become HTTP 500 with a logged message.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.models import (
    ChatRequest,
    ChatResponse,
    SkillExecutionRequest,
    SkillExecutionResponse,
    SkillRegistryEntry,
    SystemStatus,
)
from backend.orchestrator import (
    AgentOrchestrator,
    SkillNotFoundError,
    SkillValidationError,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("agentflow.api")

orchestrator: AgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create the harness when the process starts and drop it on shutdown."""
    global orchestrator
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    logger.info("Initializing harness")
    orchestrator = AgentOrchestrator(ollama_url)
    try:
        yield
    finally:
        logger.info("Shutting down harness")
        orchestrator = None


app = FastAPI(
    title="AgentFlow API",
    version="2.0.0",
    description="Open-source AI agent harness with Agent Skill Open Standard compliance",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return HTTP 400 for malformed request bodies."""
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


def get_orchestrator() -> AgentOrchestrator:
    """Return the live harness or fail if startup has not finished.

    Returns:
        The process-wide orchestrator.

    Raises:
        HTTPException: Startup has not created the harness yet.
    """
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="AgentFlow harness is not initialized.")
    return orchestrator


@app.get("/")
def root() -> dict[str, str]:
    """Welcome payload with the API version."""
    return {
        "message": "Welcome to AgentFlow, an open-source AI agent harness.",
        "version": "2.0.0",
        "docs": "/docs",
        "standard": "Agent Skill Open Standard v1.0",
    }


@app.get("/api/health", response_model=SystemStatus)
def health() -> dict[str, Any]:
    """Return Ollama connectivity, loaded skills, and process counters."""
    try:
        return get_orchestrator().get_status()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {exc}") from exc


@app.get("/api/skills", response_model=list[SkillRegistryEntry])
def skills() -> list[dict[str, Any]]:
    """Return every skill registered with the harness."""
    try:
        return get_orchestrator().list_skills()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Skill registry failed")
        raise HTTPException(status_code=500, detail=f"Could not list skills: {exc}") from exc


@app.post("/api/skills/execute", response_model=SkillExecutionResponse)
def execute_skill(body: SkillExecutionRequest) -> dict[str, Any]:
    """Run one skill and optionally enhance the result with the model."""
    harness = get_orchestrator()
    try:
        return harness.execute_skill(
            skill_name=body.skill_name,
            input_data=body.input_data,
            use_llm=body.use_llm,
            llm_prompt_template=body.llm_prompt_template,
            model=body.model,
            temperature=body.temperature,
        )
    except SkillNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SkillValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Skill execution failed")
        raise HTTPException(status_code=500, detail=f"Skill execution failed: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> dict[str, Any]:
    """Chat directly with the model, or run a skill and then enhance it."""
    harness = get_orchestrator()
    try:
        return harness.chat(
            message=body.message,
            skill_name=body.skill_name,
            model=body.model,
            temperature=body.temperature,
        )
    except SkillNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SkillValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.get("/api/metrics")
def metrics(limit: int = 50) -> dict[str, Any]:
    """Return recent execution history."""
    harness = get_orchestrator()
    history = harness.metrics.get_history(limit=limit)
    return {
        "total_requests": harness.metrics.total_requests,
        "history": history,
    }


@app.get("/api/metrics/export")
def export_metrics() -> Response:
    """Download the full metrics history as a JSON attachment."""
    payload = get_orchestrator().metrics.export_json()
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="agentflow-metrics.json"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
