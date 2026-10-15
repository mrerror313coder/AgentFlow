# 🤖 AgentFlow

**Modular AI Agent Harness — Open-Weight LLM Orchestration with Agent Skill Open Standard Compliance**

![Hacktoberfest 2026](https://img.shields.io/badge/Hacktoberfest-2026-E53927)
![MLH Hack Day Lahore](https://img.shields.io/badge/MLH-Hack%20Day%20Lahore-1B4D89)
![Built with DigitalOcean](https://img.shields.io/badge/Built%20with-DigitalOcean-0080FF)
![License: MIT](https://img.shields.io/badge/License-MIT-3D5F58)

AgentFlow is an open-source harness that runs small, typed agent skills and optionally asks an open-weight model, through [Ollama](https://ollama.com), to explain the result. Skills are discovered from a folder. Each one implements the same contract, so a new capability is a Python class rather than a new backend. If Ollama is not running, a labeled offline fallback still answers, which keeps demos, tests, and reviews working on a laptop with no GPU.

Built for Hacktoberfest Hack Day Lahore x OmniCrew. The project is aimed at **Best Open-Source AI Project** and **Build with DigitalOcean**.

## Architecture

```text
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit   │────▶│   FastAPI Server  │────▶│   Orchestrator   │
│   Dashboard   │     │   (main.py)       │     │   (Harness)      │
└──────────────┘     └──────────────────┘     └───────┬─────────┘
                                                       │
                                       ┌───────────────┼───────────────┐
                                       ▼               ▼               ▼
                                ┌────────────┐  ┌────────────┐  ┌──────────┐
                                │ Skill Hub   │  │  Ollama    │  │ Metrics  │
                                │ (4 Skills)  │  │  (Llama 3) │  │ Tracker  │
                                └────────────┘  └────────────┘  └──────────┘
                                Agent Skill
                                Open Standard
                                ✅ Compliant
```

The Streamlit app calls FastAPI when `http://localhost:8000` is up. If it is not, the same `AgentOrchestrator` runs inside the dashboard process. The orchestrator is the model harness: it validates skill input, runs the skill, records metrics, and only then sends the skill output to the model.

## Features

- 🧩 Plugin skills discovered at startup from the `skills/` folder
- 📐 Agent Skill Open Standard v1.0 reference implementation in `BaseSkill`
- 🦙 Open-weight inference through Ollama (Llama 3, Mistral, CodeLlama, Phi-3)
- 🛟 Prompt-aware offline fallback when Ollama is down or returns an empty completion
- 📝 Extractive summarizer with real sentence scoring
- 🔍 Regex code review for long functions, docstrings, bare `except`, star imports, and mutable defaults
- 🐙 GitHub pull-request diff summary, with a realistic mock diff when no token is set
- 📚 Local TF-IDF retrieval over your notes, implemented with the standard library
- 📊 Request history, token estimates, and JSON export
- 💬 Streamlit workspace with chat, a skill registry, and a metrics tab
- 🌊 DigitalOcean App Platform spec and a one-command Droplet script

## Agent Skill Open Standard compliance

Judging for this category asks for an agent skill that complies with the Agent Skill Open Standard. `skills/base_skill.py` is a reference implementation of that contract. Every bundled skill subclasses it. The registry entry on each skill includes `"standard": "Agent Skill Open Standard v1.0"`, and the Skill Hub prints that badge on every card.

| Standard requirement | Where AgentFlow implements it |
| --- | --- |
| Stable snake_case `name` | `BaseSkill.name` |
| Human-readable `description` | `BaseSkill.description` |
| JSON Schema `input_schema` with `type`, `properties`, and `required` | `BaseSkill.input_schema` |
| Semantic `version` | `BaseSkill.version` |
| `execute(input) -> {result, metadata}` | `BaseSkill.execute` |
| `metadata.execution_time_ms` and `metadata.skill_version` | `BaseSkill.build_metadata` |
| Schema checks without a hidden validator library | `BaseSkill.validate_input` |
| Registry record | `BaseSkill.to_registry_entry` |

`validate_input` checks required keys and JSON types in pure Python. It does not import `jsonschema`. Boolean values are rejected for integer fields, because `bool` is a subclass of `int` in Python.

## Open-weight AI integration

Ollama is the inference server. The dashboard model list is Llama 3, Mistral, CodeLlama, and Phi-3. Point `OLLAMA_HOST` at any machine running `ollama serve`.

`OllamaClient.smart_generate` tries `POST /api/generate` first. It falls back when the connection fails, the HTTP call errors, the JSON is invalid, or the completion is empty. The fallback reads the prompt for words such as "summarize", "review", and "code", then returns a short structured answer that quotes terms from the prompt. The response is labeled `source: "fallback"` and the model name ends in `-mock`, so a judge can see that a live model did not produce it.

That split is deliberate. Skill logic (summary scores, diff counts, TF-IDF) does not depend on a model. The model is an optional narrator. A weak or missing model cannot mark a skill as successful.

## Model harness

`backend/orchestrator.py` is the original harness.

1. **Discovery.** `skill_loader.discover_skills` imports each module under `skills/`, instantiates concrete `BaseSkill` subclasses, and registers them by `name`. A broken file is logged and skipped.
2. **Gate.** `execute_skill` refuses unknown skills and rejects input that fails the skill schema before any work runs.
3. **Execution.** The skill returns its own result. The harness times the call and estimates tokens from the text length.
4. **Context injection.** When `use_llm` is true, or when chat selects a skill, the harness renders the result into a prompt and calls `smart_generate`. The default prompt is: `Analyze the following skill output and provide insights`.
5. **Chat mapping.** A workspace message is mapped onto the skill's required fields. `rag_doc_search` receives `query`, `text_summarizer` receives `text`, `code_reviewer` receives `code`, and `owner/repo#123` fills the pull-request skill.

The harness enforces the contract in code. It does not ask the model to remember which fields are required.

## Quick start

Python 3.11 is the intended runtime. Python 3.10 or newer works for the type hints used here.

```bash
git clone https://github.com/mrerror313coder/AgentFlow.git
cd AgentFlow
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Optional, for live open-weight answers:

```bash
ollama pull llama3
ollama serve
```

Start the API and the dashboard in two terminals:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
streamlit run app.py
```

Open `http://localhost:8501`. The API docs are at `http://localhost:8000/docs`.

Try the retrieval skill with: `How does chunking reduce hallucination during retrieval?`

Without Ollama, the skill still returns the matching note and the insight panel is marked `fallback`.

## Render deployment

`render.yaml` is a Render Blueprint with two free web services. Render installs `requirements.txt`, pins Python with `.python-version`, and passes `$PORT` into both processes. The dashboard reads `AGENTFLOW_API_URL` from the API service. Ollama is not on Render, so answers use the labeled fallback.

After this repo is on GitHub, open:

https://render.com/deploy?repo=https://github.com/mrerror313coder/AgentFlow

Render connects the repo and deploys `agentflow-api` and `agentflow-ui`. Free services sleep after inactivity. The first request after sleep takes longer.

## DigitalOcean deployment

This is the **Build with DigitalOcean** path. `app.yaml` already points at `mrerror313coder/AgentFlow`.

### App Platform

1. Confirm the GitHub repository is public.
2. Set `OLLAMA_HOST` in `app.yaml` to a Droplet running Ollama, or leave the placeholder and rely on the fallback.
3. Create the app:

```bash
doctl apps create --spec app.yaml
```

The spec runs two Python services on `basic-xxs`:

- `agentflow-api` — `uvicorn backend.main:app` on port 8080
- `agentflow-ui` — Streamlit on port 8080, with `AGENTFLOW_API_URL` set to the API service's public URL

If App Platform rejects `basic-xxs` on a newer account, use `apps-s-1vcpu-0.5gb` in its place. The Python buildpack installs `requirements.txt` on its own.

### Droplet

`deploy_droplet.sh` uses [doctl](https://docs.digitalocean.com/reference/doctl/how-to/install/). Authenticate with `doctl auth init`, add an SSH key to the account, then:

```bash
export REPO_URL=https://github.com/mrerror313coder/AgentFlow.git
bash deploy_droplet.sh
```

The script creates an Ubuntu 22.04 Droplet with 2 GB of RAM when one named `agentflow` does not already exist, adds a 4 GB swap file so `llama3` can be pulled on that size, installs Python 3.11 and Ollama, clones the repo, and starts the API and dashboard with `nohup`. It prints the public IP and the URLs for ports 8000 and 8501. Ollama listens on the Droplet's localhost only.

A larger Droplet is more comfortable for Llama 3. `phi3` is the lighter model in the dashboard list if you change `OLLAMA_MODEL`.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/` | Welcome message, version `2.0.0`, standard name |
| `GET` | `/api/health` | Ollama status, loaded skills, request count, uptime |
| `GET` | `/api/skills` | Registry entries for every loaded skill |
| `POST` | `/api/skills/execute` | Run one skill. Unknown skill is 404. Invalid input is 400 |
| `POST` | `/api/chat` | Direct model chat, or skill plus LLM insight |
| `GET` | `/api/metrics` | Recent execution history |
| `GET` | `/api/metrics/export` | Download the history as JSON |

Example:

```bash
curl -X POST http://localhost:8000/api/skills/execute \
  -H "Content-Type: application/json" \
  -d "{\"skill_name\":\"text_summarizer\",\"input_data\":{\"text\":\"Agents plan. Agents act. Agents check the tool result.\"},\"use_llm\":false}"
```

Interactive docs: `http://localhost:8000/docs`.

## Create your own skill in 5 minutes

Add `skills/keyword_counter.py`:

```python
import time
from typing import Any

from skills.base_skill import BaseSkill


class KeywordCounter(BaseSkill):
    """Count how often a word appears in a passage."""

    @property
    def name(self) -> str:
        return "keyword_counter"

    @property
    def description(self) -> str:
        return "Count case-insensitive occurrences of a keyword in text."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "keyword": {"type": "string"},
            },
            "required": ["text", "keyword"],
        }

    @property
    def version(self) -> str:
        return "1.0.0"

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.ensure_valid(input_data)
        started = time.perf_counter()
        text = str(input_data["text"])
        keyword = str(input_data["keyword"])
        count = text.lower().count(keyword.lower())
        return {
            "result": {"keyword": keyword, "count": count},
            "metadata": self.build_metadata(started, characters=len(text)),
        }
```

Restart the API or the dashboard. The loader picks the class up by subclassing `BaseSkill`. No registry edit is required.

## Bundled skills

| Name | What it does |
| --- | --- |
| `text_summarizer` | Scores sentences by word frequency and keeps the top ones in original order |
| `code_reviewer` | Static checks for long functions, missing docstrings, bare `except`, star imports, and mutable defaults |
| `github_pr_summarizer` | Reads a pull request file list from GitHub, or builds a mock diff and counts `+` / `-` lines |
| `rag_doc_search` | Chunks `data/sample_notes`, builds TF-IDF vectors, and ranks chunks by cosine similarity |

Sample notes cover language models, agent loops, and retrieval. Replace them with your own `.txt` or `.md` files and pass `data_dir` to the retrieval skill.

The pull-request skill uses the live GitHub API only when `github_token` is included in the skill input. Leave it blank to stay in mock mode. Do not commit tokens. `.env.example` shows `GITHUB_TOKEN` as an empty reminder, and `.gitignore` excludes `.env`.

## Tech stack

Python, FastAPI, Pydantic v2, Streamlit, Requests, Ollama, and DigitalOcean App Platform plus Droplets. There is no scikit-learn, NumPy, LangChain, or hosted-model SDK. TF-IDF and sentence scoring are ordinary Python so the skills stay readable in a one-day build.

## Testing

```bash
pytest tests/ -q
```

`tests/test_skills.py` checks validation, summary order, review findings, mock diffs, and retrieval ranking. `tests/test_orchestrator.py` checks discovery and the success and failure paths. `tests/test_api.py` checks HTTP 200, 400, and 404 with FastAPI's test client. Ollama is not required. The tests point the client at a closed port and expect `source: "fallback"`.

## Contributing

1. Fork the repository and create a branch.
2. Add a skill by subclassing `BaseSkill`, or fix a harness bug with a test.
3. Run `pytest tests/ -q`.
4. Open a pull request that explains why the change helps a caller of the harness.

Keep new skills free of machine-learning dependencies unless you also add them to `requirements.txt` and explain the reason. Match the existing type hints and docstrings. Do not commit `.env` files or GitHub tokens.

## License

MIT. Copyright 2026 AgentFlow Contributors. See [LICENSE](LICENSE).

## Hackathon submission checklist

- ✅ Public repository (push this tree and flip it to public)
- ✅ MIT license
- ✅ Open-weight AI through Ollama, with an honest offline fallback
- ✅ Agent Skill Open Standard compliance in `BaseSkill` and every bundled skill
- ✅ Original model harness in `backend/orchestrator.py`
- ✅ DigitalOcean deploy via `app.yaml` and `deploy_droplet.sh`
- ✅ Tests for skills, the harness, and the HTTP API
- ✅ Sample notes so retrieval works on a fresh clone

AI use: the harness, skills, dashboard, tests, and this README were drafted with an AI coding assistant in Cursor and then checked by running the test suite. Live model answers, when Ollama is running, come from the open-weight model you select. Offline answers are the deterministic fallback and are labeled as such.
