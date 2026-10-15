"""Static-analysis code review skill.

Flags a small, explainable set of Python issues with regular expressions:
long functions, missing docstrings, bare except handlers, star imports, and
mutable default arguments. Findings are deterministic and do not call a model.
"""

from __future__ import annotations

import re
import time
from typing import Any

from skills.base_skill import BaseSkill

_DEF_OR_CLASS = re.compile(r"^\s*(async\s+def\s+|def\s+|class\s+)")
_FUNCTION = re.compile(r"^\s*(async\s+def\s+|def\s+)")
_STAR_IMPORT = re.compile(r"from\s+.+\s+import\s+\*")
_BARE_EXCEPT = re.compile(r"^\s*except\s*:")
_MUTABLE_DEFAULT = re.compile(r"=\s*(\[\]|\{\}|set\(\))")
_LONG_FUNCTION_LINES = 20


class CodeReviewer(BaseSkill):
    """Review source code for a fixed set of maintainability rules."""

    @property
    def name(self) -> str:
        """Registry id for the code reviewer."""
        return "code_reviewer"

    @property
    def description(self) -> str:
        """Short description shown in the skill registry."""
        return (
            "Regex static analysis that flags long functions, missing "
            "docstrings, bare excepts, star imports, and mutable defaults."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for code review input."""
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Source code to review.",
                },
                "language": {
                    "type": "string",
                    "description": "Language hint. Rules target Python syntax.",
                },
            },
            "required": ["code"],
        }

    @property
    def version(self) -> str:
        """Semantic version of the reviewer."""
        return "1.0.0"

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the static checks and return structured findings.

        Args:
            input_data: Must include ``code``. ``language`` is optional.

        Returns:
            A result object with ``findings`` and a one-line ``summary``.

        Raises:
            ValueError: If the input does not match ``input_schema``.
        """
        self.ensure_valid(input_data)
        started = time.perf_counter()
        code = str(input_data["code"])
        language = str(input_data.get("language") or "python")
        lines = code.splitlines()
        findings: list[dict[str, Any]] = []

        for line_number, line in enumerate(lines, start=1):
            if _STAR_IMPORT.search(line):
                findings.append(
                    self._finding(
                        line_number,
                        "star_import",
                        "warning",
                        "Star import hides names and makes reviews harder to trace.",
                    )
                )
            if _BARE_EXCEPT.search(line):
                findings.append(
                    self._finding(
                        line_number,
                        "bare_except",
                        "error",
                        "Bare except catches everything, including KeyboardInterrupt.",
                    )
                )
            if _FUNCTION.match(line) and _MUTABLE_DEFAULT.search(line):
                findings.append(
                    self._finding(
                        line_number,
                        "mutable_default",
                        "error",
                        "Mutable default argument is shared across calls.",
                    )
                )

        headers = [
            index
            for index, line in enumerate(lines)
            if _DEF_OR_CLASS.match(line)
        ]
        for position, start in enumerate(headers):
            end = headers[position + 1] if position + 1 < len(headers) else len(lines)
            header = lines[start]
            if _FUNCTION.match(header) and (end - start) > _LONG_FUNCTION_LINES:
                findings.append(
                    self._finding(
                        start + 1,
                        "long_function",
                        "warning",
                        f"Function spans {end - start} lines, which is over {_LONG_FUNCTION_LINES}.",
                    )
                )
            if not self._has_docstring(lines, start, end):
                kind = "function" if _FUNCTION.match(header) else "class"
                findings.append(
                    self._finding(
                        start + 1,
                        "missing_docstring",
                        "info",
                        f"This {kind} has no docstring on the following non-empty line.",
                    )
                )

        findings.sort(key=lambda item: (item["line"], item["rule"]))
        counts = {"error": 0, "warning": 0, "info": 0}
        for finding in findings:
            severity = str(finding["severity"])
            counts[severity] = counts.get(severity, 0) + 1
        summary = (
            f"Reviewed {len(lines)} line(s) and found {len(findings)} issue(s): "
            f"{counts['error']} error(s), {counts['warning']} warning(s), "
            f"{counts['info']} info."
        )
        return {
            "result": {"findings": findings, "summary": summary},
            "metadata": self.build_metadata(
                started,
                language=language,
                lines=len(lines),
                finding_count=len(findings),
                rules_target="python",
            ),
        }

    @staticmethod
    def _finding(line: int, rule: str, severity: str, message: str) -> dict[str, Any]:
        """Build one finding record."""
        return {
            "line": line,
            "rule": rule,
            "severity": severity,
            "message": message,
        }

    @staticmethod
    def _has_docstring(lines: list[str], start: int, end: int) -> bool:
        """Return whether a def/class block opens with a docstring."""
        header = lines[start]
        if '"""' in header or "'''" in header:
            return True
        for index in range(start + 1, end):
            stripped = lines[index].strip()
            if not stripped:
                continue
            return stripped.startswith('"""') or stripped.startswith("'''")
        return False
