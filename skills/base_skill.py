"""Agent Skill Open Standard — reference implementation.

Every AgentFlow skill implements this interface so tools stay interchangeable:

* Discoverability via ``name``, ``description``, and ``version``.
* Schema-driven input validation via a JSON Schema ``input_schema``.
* A deterministic execution contract: ``execute`` returns ``result`` and ``metadata``.
* Registry compatibility through ``to_registry_entry``.

This module is the compliance boundary for Agent Skill Open Standard v1.0.
Validation is implemented in pure Python so skills do not depend on jsonschema.
"""

from __future__ import annotations

import abc
from typing import Any


JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

STANDARD_NAME = "Agent Skill Open Standard v1.0"


class BaseSkill(abc.ABC):
    """Abstract skill contract shared by every AgentFlow capability.

    Subclasses must expose identity metadata and an ``execute`` method that
    returns ``{"result": ..., "metadata": {...}}``. The metadata dictionary
    always includes ``execution_time_ms`` and ``skill_version``.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier, for example ``rag_doc_search``."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """One or two sentence human-readable description of the skill."""

    @property
    @abc.abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema object describing accepted ``input_data`` keys."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Semantic version string for this skill implementation."""

    @abc.abstractmethod
    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the skill.

        Args:
            input_data: Arguments matching ``input_schema``.

        Returns:
            A dictionary with ``result`` and ``metadata`` keys. ``metadata``
            includes ``execution_time_ms`` and ``skill_version``.

        Raises:
            ValueError: If ``input_data`` does not satisfy the schema.
        """

    def validate_input(self, input_data: dict[str, Any]) -> tuple[bool, str]:
        """Validate ``input_data`` against this skill's JSON Schema.

        Required keys must be present. When a property declares a JSON
        ``type``, the Python value must match that type. Unknown keys are
        allowed so callers can attach optional context. Boolean values are
        rejected for ``integer`` and ``number`` fields because ``bool`` is a
        subclass of ``int`` in Python.

        Args:
            input_data: Candidate skill arguments.

        Returns:
            ``(True, "")`` when the payload is valid, otherwise
            ``(False, "error message")``.
        """
        if not isinstance(input_data, dict):
            return False, "input_data must be a JSON object."

        schema = self.input_schema
        if schema.get("type") not in (None, "object"):
            return False, "Skill input_schema must describe a JSON object."

        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        if not isinstance(properties, dict):
            return False, "Skill input_schema properties must be an object."
        if not isinstance(required, list):
            return False, "Skill input_schema required must be an array."

        missing = [key for key in required if key not in input_data]
        if missing:
            joined = ", ".join(str(key) for key in missing)
            return False, f"Missing required field(s): {joined}."

        for key, value in input_data.items():
            prop = properties.get(key)
            if not isinstance(prop, dict):
                continue
            expected = prop.get("type")
            if expected is None:
                continue
            if isinstance(expected, list):
                allowed = expected
            else:
                allowed = [expected]
            if not any(self._value_matches(value, json_type) for json_type in allowed):
                readable = " or ".join(str(item) for item in allowed)
                return False, (
                    f"Field '{key}' must be of type {readable}, "
                    f"got {type(value).__name__}."
                )
        return True, ""

    def to_registry_entry(self) -> dict[str, Any]:
        """Serialize this skill for the public registry endpoint.

        Returns:
            Identity, schema, version, and the standard name judges and
            clients use to confirm compliance.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "version": self.version,
            "standard": STANDARD_NAME,
        }

    def ensure_valid(self, input_data: dict[str, Any]) -> None:
        """Raise ``ValueError`` when ``input_data`` fails validation.

        Args:
            input_data: Candidate skill arguments.

        Raises:
            ValueError: The payload is missing a required field or has a
                value of the wrong JSON type.
        """
        ok, message = self.validate_input(input_data)
        if not ok:
            raise ValueError(message)

    def build_metadata(self, started: float, **extra: Any) -> dict[str, Any]:
        """Build the metadata object required by the execution contract.

        Args:
            started: Value previously returned by ``time.perf_counter``.
            **extra: Skill-specific metadata fields.

        Returns:
            Metadata including elapsed milliseconds and ``skill_version``.
        """
        import time

        payload: dict[str, Any] = {
            "execution_time_ms": round((time.perf_counter() - started) * 1000, 3),
            "skill_version": self.version,
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _value_matches(value: Any, json_type: str) -> bool:
        """Return whether ``value`` matches a single JSON Schema type name."""
        if json_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if json_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        expected = JSON_TYPE_MAP.get(json_type)
        if expected is None:
            return True
        return isinstance(value, expected)
