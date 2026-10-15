"""GitHub pull-request diff summarizer.

When a token is supplied, the skill reads the pull request file list from the
GitHub REST API. If the token is missing, or GitHub rejects the call, it
builds a realistic unified diff locally and summarizes that instead. Diff
statistics are counted from ``+`` and ``-`` lines so the narrative matches
the patch text.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from skills.base_skill import BaseSkill

logger = logging.getLogger("agentflow.skills.github_pr_summarizer")

_GITHUB_ACCEPT = "application/vnd.github.v3+json"
_TIMEOUT_SECONDS = 20


class GitHubPRSummarizer(BaseSkill):
    """Summarize additions, deletions, and intent of a pull request."""

    @property
    def name(self) -> str:
        """Registry id for the pull-request summarizer."""
        return "github_pr_summarizer"

    @property
    def description(self) -> str:
        """Short description shown in the skill registry."""
        return (
            "Summarize a GitHub pull request diff from the REST API, "
            "with a realistic mock diff when no token is available."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for pull-request summarizer input."""
        return {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in owner/name form.",
                },
                "pr_number": {
                    "type": "integer",
                    "description": "Pull request number.",
                },
                "github_token": {
                    "type": "string",
                    "description": "Optional GitHub token. Omit it to use mock mode.",
                },
            },
            "required": ["repo", "pr_number"],
        }

    @property
    def version(self) -> str:
        """Semantic version of the pull-request summarizer."""
        return "1.0.0"

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize one pull request.

        Args:
            input_data: ``repo`` and ``pr_number`` are required. ``github_token``
                selects the live GitHub API when it is a non-empty string.

        Returns:
            File stats, totals, and a short narrative paragraph.

        Raises:
            ValueError: If the input does not match ``input_schema``.
        """
        self.ensure_valid(input_data)
        started = time.perf_counter()
        repo = str(input_data["repo"]).strip()
        pr_number = int(input_data["pr_number"])
        token = str(input_data.get("github_token") or "").strip()

        source = "mock"
        notice = ""
        raw_files: list[dict[str, Any]]
        if token:
            fetched, notice = self._fetch_files(repo, pr_number, token)
            if fetched is None:
                raw_files = self._mock_files()
                source = "mock"
            else:
                raw_files = fetched
                source = "github_api"
        else:
            raw_files = self._mock_files()
            notice = "No GitHub token was provided, so a simulated diff was used."

        files = [self._summarize_file(item) for item in raw_files]
        total_additions = sum(item["additions"] for item in files)
        total_deletions = sum(item["deletions"] for item in files)
        narrative = self._narrative(files, total_additions, total_deletions)
        return {
            "result": {
                "total_files": len(files),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "files": files,
                "narrative": narrative,
            },
            "metadata": self.build_metadata(
                started,
                source=source,
                repo=repo,
                pr_number=pr_number,
                notice=notice,
            ),
        }

    def _fetch_files(
        self,
        repo: str,
        pr_number: int,
        token: str,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Download the pull request file list.

        Args:
            repo: ``owner/name`` repository slug.
            pr_number: Pull request number.
            token: Bearer token. It is sent to GitHub and never logged.

        Returns:
            ``(files, "")`` on success, or ``(None, reason)`` when the API
            cannot be used and the caller should fall back to mock mode.
        """
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": _GITHUB_ACCEPT,
            "User-Agent": "AgentFlow",
        }
        collected: list[dict[str, Any]] = []
        try:
            for page in range(1, 6):
                response = requests.get(
                    url,
                    headers=headers,
                    params={"per_page": 100, "page": page},
                    timeout=_TIMEOUT_SECONDS,
                )
                if response.status_code in {401, 403, 404}:
                    message = (
                        f"GitHub API returned {response.status_code} for "
                        f"{repo}#{pr_number}; using a simulated diff."
                    )
                    logger.warning(message)
                    return None, message
                if response.status_code != 200:
                    message = (
                        f"GitHub API returned {response.status_code}; "
                        "using a simulated diff."
                    )
                    logger.warning(message)
                    return None, message
                payload = response.json()
                if not isinstance(payload, list):
                    message = "GitHub API returned an unexpected payload; using a simulated diff."
                    logger.warning(message)
                    return None, message
                collected.extend(item for item in payload if isinstance(item, dict))
                if len(payload) < 100:
                    break
        except requests.RequestException as exc:
            message = f"GitHub API request failed ({exc}); using a simulated diff."
            logger.warning(message)
            return None, message
        except ValueError as exc:
            message = f"GitHub API returned invalid JSON ({exc}); using a simulated diff."
            logger.warning(message)
            return None, message
        logger.info("Fetched %d file(s) for %s#%s", len(collected), repo, pr_number)
        return collected, ""

    @staticmethod
    def _mock_files() -> list[dict[str, Any]]:
        """Return a four-file pull request that looks like a real auth change."""
        return [
            {
                "filename": "src/auth/login.py",
                "status": "modified",
                "patch": (
                    "@@ -10,6 +10,12 @@ def login(user, password):\n"
                    "     if not user:\n"
                    "-        return None\n"
                    "+        raise ValueError(\"user is required\")\n"
                    "+    if not password:\n"
                    "+        raise ValueError(\"password is required\")\n"
                    "     token = issue_token(user)\n"
                    "+    audit_log(user, \"login\")\n"
                    "     return token\n"
                ),
            },
            {
                "filename": "src/models/user.py",
                "status": "added",
                "patch": (
                    "@@ -0,0 +1,14 @@\n"
                    "+class User:\n"
                    "+    \"\"\"Account record used by the login flow.\"\"\"\n"
                    "+\n"
                    "+    def __init__(self, email: str) -> None:\n"
                    "+        self.email = email\n"
                    "+        self.failed_attempts = 0\n"
                    "+\n"
                    "+    def record_failure(self) -> None:\n"
                    "+        self.failed_attempts += 1\n"
                ),
            },
            {
                "filename": "tests/test_auth.py",
                "status": "modified",
                "patch": (
                    "@@ -4,3 +4,9 @@ def test_login_success():\n"
                    "     assert login(\"ada\", \"secret\").startswith(\"tok_\")\n"
                    "+\n"
                    "+def test_login_requires_password():\n"
                    "+    with pytest.raises(ValueError):\n"
                    "+        login(\"ada\", \"\")\n"
                ),
            },
            {
                "filename": "README.md",
                "status": "modified",
                "patch": (
                    "@@ -22,2 +22,4 @@\n"
                    " ## Authentication\n"
                    "+Login now rejects empty passwords and writes an audit event.\n"
                    " See `src/auth/login.py`.\n"
                ),
            },
        ]

    def _summarize_file(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert one GitHub file object or mock file into summary stats."""
        filename = str(item.get("filename") or "unknown")
        patch = item.get("patch") or ""
        if isinstance(patch, str) and patch:
            additions, deletions = self._count_diff(patch)
        else:
            additions = int(item.get("additions") or 0)
            deletions = int(item.get("deletions") or 0)
        change_type = str(item.get("status") or self._infer_change_type(additions, deletions))
        return {
            "filename": filename,
            "additions": additions,
            "deletions": deletions,
            "change_type": change_type,
        }

    @staticmethod
    def _count_diff(patch: str) -> tuple[int, int]:
        """Count added and deleted lines in a unified diff snippet."""
        additions = 0
        deletions = 0
        for line in patch.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                deletions += 1
        return additions, deletions

    @staticmethod
    def _infer_change_type(additions: int, deletions: int) -> str:
        """Infer a change type when the API did not provide ``status``."""
        if additions and not deletions:
            return "added"
        if deletions and not additions:
            return "removed"
        return "modified"

    def _narrative(
        self,
        files: list[dict[str, Any]],
        total_additions: int,
        total_deletions: int,
    ) -> str:
        """Build the one-paragraph summary required by the skill contract."""
        categories = self._join_phrases([self._category(item["filename"]) for item in files])
        if not files:
            return (
                "This PR modifies 0 files across the repository. "
                f"Overall: +{total_additions}/-{total_deletions} lines net."
            )
        largest = max(files, key=lambda item: item["additions"] + item["deletions"])
        return (
            f"This PR modifies {len(files)} files across {categories}. "
            f"The largest change is in {largest['filename']} with "
            f"+{largest['additions']}/-{largest['deletions']} lines. "
            f"Overall: +{total_additions}/-{total_deletions} lines net."
        )

    @staticmethod
    def _category(filename: str) -> str:
        """Group a path by its top-level directory, or by a docs/root bucket."""
        parts = filename.split("/")
        if len(parts) > 1 and parts[0]:
            return parts[0]
        if filename.lower().endswith(".md"):
            return "docs"
        return "root"

    @staticmethod
    def _join_phrases(items: list[str]) -> str:
        """Join unique category names with commas and a final 'and'."""
        unique: list[str] = list(dict.fromkeys(items))
        if not unique:
            return "the repository"
        if len(unique) == 1:
            return unique[0]
        if len(unique) == 2:
            return f"{unique[0]} and {unique[1]}"
        return ", ".join(unique[:-1]) + f", and {unique[-1]}"
