"""AgentFlow model harness.

``AgentOrchestrator`` is the original harness entry point. It discovers
skills that implement the Agent Skill Open Standard, executes them behind a
validation gate, and optionally chains the structured result into an
open-weight model served by Ollama. The model never decides whether a skill
ran successfully: the skill return value and the metrics tracker do.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from backend.metrics import MetricsTracker
from backend.ollama_client import OllamaClient
from backend.skill_loader import discover_skills, get_registry
from skills.base_skill import BaseSkill

logger = logging.getLogger("agentflow.orchestrator")

_DEFAULT_TEMPLATE = "Analyze the following skill output and provide insights:\n\n{result}"
_PR_PATTERN = re.compile(r"([\w.-]+/[\w.-]+)\s*#?\s*(\d+)")


class SkillNotFoundError(KeyError):
    """Raised when a caller asks for a skill that is not registered."""


class SkillValidationError(ValueError):
    """Raised when skill input does not match its JSON Schema."""


class AgentOrchestrator:
    """Chain registered skills with optional open-weight LLM inference."""

    def __init__(self, ollama_url: str) -> None:
        """Connect the Ollama client, metrics tracker, and skill registry.

        Args:
            ollama_url: Base URL of the Ollama server.
        """
        logger.info("Starting AgentFlow harness with Ollama at %s", ollama_url)
        self.client = OllamaClient(ollama_url)
        self.metrics = MetricsTracker()
        try:
            self.skills = discover_skills()
        except FileNotFoundError as exc:
            logger.error("Skill discovery failed: %s", exc)
            self.skills = {}
        logger.info(
            "Harness ready with skills: %s",
            ", ".join(sorted(self.skills)) or "(none)",
        )

    def list_skills(self) -> list[dict[str, Any]]:
        """Return registry entries for every loaded skill.

        Returns:
            Dictionaries produced by ``BaseSkill.to_registry_entry``.
        """
        return get_registry(self.skills)

    def execute_skill(
        self,
        skill_name: str,
        input_data: dict[str, Any],
        use_llm: bool,
        llm_prompt_template: str | None,
        model: str,
        temperature: float,
    ) -> dict[str, Any]:
        """Validate and run one skill, optionally asking the model to comment.

        Args:
            skill_name: Registered skill name.
            input_data: Skill arguments.
            use_llm: When true, send the rendered result to ``smart_generate``.
            llm_prompt_template: Template containing ``{result}``. A default
                analysis prompt is used when this is omitted.
            model: Ollama model name.
            temperature: Sampling temperature for the model call.

        Returns:
            Skill name, raw result, metrics, and any model commentary.

        Raises:
            SkillNotFoundError: ``skill_name`` is not registered.
            SkillValidationError: ``input_data`` fails the skill schema.
        """
        skill = self._require_skill(skill_name)
        self._validate(skill, input_data)
        started = time.perf_counter()
        output = skill.execute(input_data)
        result = output.get("result")
        llm_response: str | None = None
        llm_source: str | None = None
        model_used: str | None = None
        if use_llm:
            prompt = self._build_prompt(llm_prompt_template, result)
            generated = self.client.smart_generate(prompt, model=model, temperature=temperature)
            llm_response = str(generated.get("response") or "")
            llm_source = str(generated.get("source") or "")
            model_used = str(generated.get("model") or model)
        elapsed = (time.perf_counter() - started) * 1000
        blob = self._render_result(result)
        if llm_response:
            blob = f"{blob}\n{llm_response}"
        metrics = self._store_metrics(
            skill_name=skill.name,
            elapsed_ms=elapsed,
            text=blob,
            model_used=model_used,
            skill_version=skill.version,
        )
        logger.info(
            "Executed skill=%s llm=%s elapsed_ms=%.1f",
            skill.name,
            use_llm,
            elapsed,
        )
        return {
            "skill_name": skill.name,
            "result": result,
            "metrics": metrics,
            "llm_response": llm_response,
            "llm_source": llm_source,
        }

    def chat(
        self,
        message: str,
        skill_name: str | None,
        model: str,
        temperature: float,
    ) -> dict[str, Any]:
        """Answer a chat turn, optionally running a skill first.

        A named skill is filled from the message, executed, and then enhanced
        by the model. Without a skill, the message goes straight to the model.

        Args:
            message: User text from the dashboard.
            skill_name: Optional registered skill to run before the model.
            model: Ollama model name.
            temperature: Sampling temperature.

        Returns:
            The assistant text, the skill that ran, and execution metrics.

        Raises:
            SkillNotFoundError: ``skill_name`` is not registered.
            SkillValidationError: The mapped input fails the skill schema.
        """
        started = time.perf_counter()
        if skill_name:
            skill = self._require_skill(skill_name)
            payload = self.map_message_to_input(skill, message)
            self._validate(skill, payload)
            output = skill.execute(payload)
            rendered = self._render_result(output.get("result"))
            prompt = self._build_prompt(None, output.get("result"))
            generated = self.client.smart_generate(prompt, model=model, temperature=temperature)
            source = str(generated.get("source") or "fallback")
            insight = str(generated.get("response") or "")
            response = (
                f"{rendered}\n\n---\n\n**LLM insight** ({source})\n\n{insight}"
            )
            model_used = str(generated.get("model") or model)
            version = skill.version
            used = skill.name
        else:
            generated = self.client.smart_generate(message, model=model, temperature=temperature)
            response = str(generated.get("response") or "")
            model_used = str(generated.get("model") or model)
            version = "n/a"
            used = None
        elapsed = (time.perf_counter() - started) * 1000
        metrics = self._store_metrics(
            skill_name=used or "direct_chat",
            elapsed_ms=elapsed,
            text=response,
            model_used=model_used,
            skill_version=version,
        )
        logger.info("Chat turn skill=%s elapsed_ms=%.1f", used or "direct_chat", elapsed)
        return {"response": response, "skill_used": used, "metrics": metrics}

    def get_status(self) -> dict[str, Any]:
        """Return process status, including a live Ollama health check.

        Returns:
            A dictionary matching the ``SystemStatus`` schema.
        """
        try:
            ok, models = self.client.health_check()
        except Exception as exc:
            logger.warning("Status health check failed: %s", exc)
            ok, models = False, []
        status = self.metrics.get_status(ok, sorted(self.skills))
        status["ollama_models"] = models
        status["ollama_connected"] = ok
        return status

    def map_message_to_input(self, skill: BaseSkill, message: str) -> dict[str, Any]:
        """Map a free-text chat message onto a skill's required fields.

        Pull request messages of the form ``owner/repo#123`` fill both required
        fields. Every other skill receives the message in its first required
        string field.

        Args:
            skill: Skill that will receive the mapped payload.
            message: Raw user message.

        Returns:
            An ``input_data`` dictionary for ``skill.execute``.
        """
        if skill.name == "github_pr_summarizer":
            match = _PR_PATTERN.search(message)
            if match:
                return {"repo": match.group(1), "pr_number": int(match.group(2))}
            return {"repo": "octocat/Hello-World", "pr_number": 1}

        schema = skill.input_schema
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        payload: dict[str, Any] = {}
        for key in required:
            prop = properties.get(key) or {}
            json_type = prop.get("type")
            if json_type == "string":
                payload[key] = message
            elif json_type == "integer":
                numbers = re.findall(r"\d+", message)
                payload[key] = int(numbers[0]) if numbers else 1
        if payload:
            return payload
        for key, prop in properties.items():
            if isinstance(prop, dict) and prop.get("type") == "string":
                return {key: message}
        return {"input": message}

    def _require_skill(self, skill_name: str) -> BaseSkill:
        """Return a skill or raise ``SkillNotFoundError``."""
        skill = self.skills.get(skill_name)
        if skill is None:
            known = ", ".join(sorted(self.skills)) or "(none)"
            raise SkillNotFoundError(f"Skill '{skill_name}' is not registered. Known skills: {known}.")
        return skill

    @staticmethod
    def _validate(skill: BaseSkill, input_data: dict[str, Any]) -> None:
        """Raise ``SkillValidationError`` when schema validation fails."""
        ok, message = skill.validate_input(input_data)
        if not ok:
            raise SkillValidationError(message)

    @staticmethod
    def _build_prompt(template: str | None, result: Any) -> str:
        """Insert the rendered skill result into the LLM prompt template."""
        rendered = AgentOrchestrator._render_result(result)
        chosen = template or _DEFAULT_TEMPLATE
        if "{result}" in chosen:
            return chosen.replace("{result}", rendered)
        return f"{chosen}\n\n{rendered}"

    @staticmethod
    def _render_result(result: Any) -> str:
        """Turn a skill result into text the dashboard and model can read."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "narrative" in result:
            lines = [str(result["narrative"]), ""]
            for item in result.get("files") or []:
                lines.append(
                    f"- `{item.get('filename')}` ({item.get('change_type')}): "
                    f"+{item.get('additions')}/-{item.get('deletions')}"
                )
            return "\n".join(lines).strip()
        if isinstance(result, dict) and "findings" in result:
            lines = [str(result.get("summary") or "Code review")]
            for finding in result.get("findings") or []:
                lines.append(
                    f"- L{finding.get('line')} [{finding.get('severity')}] "
                    f"{finding.get('rule')}: {finding.get('message')}"
                )
            return "\n".join(lines)
        if isinstance(result, dict) and "matches" in result:
            lines = [f"Query: {result.get('query', '')}"]
            for match in result.get("matches") or []:
                score = match.get("score", 0)
                lines.append(
                    f"\n**{match.get('source')}** (chunk {match.get('chunk_index')}, "
                    f"score {score})\n\n{match.get('text', '')}"
                )
            if not result.get("matches"):
                lines.append("\nNo matching chunks were found.")
            return "\n".join(lines).strip()
        try:
            return json.dumps(result, indent=2, default=str)
        except TypeError:
            return str(result)

    def _store_metrics(
        self,
        skill_name: str,
        elapsed_ms: float,
        text: str,
        model_used: str | None,
        skill_version: str,
    ) -> dict[str, Any]:
        """Record a call and return the public metrics object."""
        tokens = self.metrics.estimate_tokens(text)
        self.metrics.record(skill_name, elapsed_ms, tokens, model_used)
        return {
            "execution_time_ms": round(elapsed_ms, 3),
            "estimated_tokens": tokens,
            "model_used": model_used,
            "skill_version": skill_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
