"""AgentFlow Streamlit dashboard.

The dashboard prefers the FastAPI harness at ``AGENTFLOW_API_URL``. When that
server is down, it imports ``AgentOrchestrator`` and runs the same skills in
process. Chat history stays in session state and can be exported as JSON or
Markdown.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from backend.ollama_client import same_origin

load_dotenv()

st.set_page_config(page_title="AgentFlow", page_icon="🤖", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
API_BASE = os.getenv("AGENTFLOW_API_URL", "http://localhost:8000").rstrip("/")
_CONFIGURED_OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
DEFAULT_OLLAMA = (
    "http://127.0.0.1:11434"
    if same_origin(_CONFIGURED_OLLAMA, API_BASE)
    else _CONFIGURED_OLLAMA
)
OLLAMA_OFFLINE = (
    "Ollama is not running on this Render service. Direct Chat and "
    "Enhance with LLM still work, and each answer is labeled source: fallback."
)
MODELS = ["llama3", "mistral", "codellama", "phi3"]

st.markdown(
    """
    <style>
    .stApp {
        background: #0f1117;
        color: #e8e6e3;
    }
    .af-card {
        background: #1b1e27;
        border: 1px solid #313644;
        border-radius: 14px;
        padding: 1rem 1.15rem 0.9rem 1.15rem;
        margin-bottom: 0.9rem;
    }
    .af-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f5f3ef;
        margin-bottom: 0.25rem;
    }
    .af-desc {
        color: #c8c6c2;
        margin-bottom: 0.6rem;
    }
    .af-badge {
        display: inline-block;
        background: #1b4332;
        color: #95d5b2;
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }
    .af-version {
        display: inline-block;
        background: #1d3557;
        color: #a8dadc;
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
    }
    .af-mode-api {
        background: #1b4332;
        color: #d8f3dc;
        border-radius: 10px;
        padding: 0.45rem 0.7rem;
        font-weight: 600;
    }
    .af-mode-local {
        background: #5c4d1a;
        color: #fff3bf;
        border-radius: 10px;
        padding: 0.45rem 0.7rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    """Create session keys used by the chat workspace and connection probe."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("ollama_probe", None)
    st.session_state.setdefault("mode", "checking")


def api_reachable(base_url: str) -> bool:
    """Return whether the FastAPI health endpoint answers."""
    try:
        response = requests.get(f"{base_url}/api/health", timeout=8)
        return response.status_code == 200
    except requests.RequestException:
        return False


@st.cache_resource
def local_orchestrator(ollama_url: str) -> Any:
    """Build one in-process harness for the selected Ollama URL."""
    from backend.orchestrator import AgentOrchestrator

    return AgentOrchestrator(ollama_url)


def probe_ollama(url: str) -> tuple[bool, list[str], str]:
    """Ask Ollama for installed models from the dashboard machine.

    Args:
        url: Ollama origin typed in the sidebar.

    Returns:
        Reachability, model names, and a short status sentence.
    """
    cleaned = url.strip().rstrip("/")
    if same_origin(cleaned, API_BASE):
        return (
            False,
            [],
            "That address is the AgentFlow API, not an Ollama server. " + OLLAMA_OFFLINE,
        )
    try:
        response = requests.get(f"{cleaned}/api/tags", timeout=3)
        response.raise_for_status()
        payload = response.json()
        names = [
            str(item.get("name"))
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        ]
        return True, names, f"Connected. {len(names)} model(s) installed."
    except requests.RequestException as exc:
        if os.getenv("RENDER"):
            return False, [], OLLAMA_OFFLINE
        return False, [], f"Could not reach Ollama: {exc}"
    except ValueError as exc:
        return False, [], f"Ollama returned invalid JSON: {exc}"


def fetch_skills(use_api: bool, orchestrator: Any) -> list[dict[str, Any]]:
    """Load the skill registry from the API or the local harness."""
    if use_api:
        try:
            response = requests.get(f"{API_BASE}/api/skills", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.RequestException as exc:
            st.error(f"Could not load skills from the API: {exc}")
            return []
    return orchestrator.list_skills()


def fetch_status(use_api: bool, orchestrator: Any) -> dict[str, Any]:
    """Load system status from the API or the local harness."""
    if use_api:
        try:
            response = requests.get(f"{API_BASE}/api/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except requests.RequestException as exc:
            return {"error": str(exc)}
    return orchestrator.get_status()


def fetch_history(use_api: bool, orchestrator: Any) -> list[dict[str, Any]]:
    """Load recent execution history."""
    if use_api:
        try:
            response = requests.get(f"{API_BASE}/api/metrics", timeout=5)
            response.raise_for_status()
            data = response.json()
            history = data.get("history", []) if isinstance(data, dict) else []
            return history if isinstance(history, list) else []
        except requests.RequestException as exc:
            st.error(f"Could not load metrics: {exc}")
            return []
    return orchestrator.metrics.get_history()


def run_turn(
    message: str,
    skill_name: str | None,
    enhance: bool,
    model: str,
    temperature: float,
    use_api: bool,
    orchestrator: Any,
) -> dict[str, Any]:
    """Execute one workspace turn and normalize it for the chat transcript.

    Args:
        message: Text from the chat box.
        skill_name: Selected skill, or None for direct chat.
        enhance: When true, a skill result is passed through the model.
        model: Model name from the sidebar.
        temperature: Sampling temperature from the sidebar.
        use_api: Whether to call FastAPI.
        orchestrator: Local harness used when the API is down.

    Returns:
        Assistant content, metrics, and the skill name that ran.
    """
    if skill_name and not enhance:
        return _run_skill_only(message, skill_name, model, temperature, use_api, orchestrator)
    if use_api:
        try:
            response = requests.post(
                f"{API_BASE}/api/chat",
                json={
                    "message": message,
                    "skill_name": skill_name,
                    "model": model,
                    "temperature": temperature,
                },
                timeout=180,
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text) if response.content else response.text
                return {
                    "content": f"Request failed ({response.status_code}): {detail}",
                    "metrics": {},
                    "skill": skill_name,
                }
            body = response.json()
            return {
                "content": str(body.get("response") or ""),
                "metrics": body.get("metrics") or {},
                "skill": body.get("skill_used"),
            }
        except requests.RequestException as exc:
            return {"content": f"API request failed: {exc}", "metrics": {}, "skill": skill_name}
    output = orchestrator.chat(message, skill_name, model, temperature)
    return {
        "content": output["response"],
        "metrics": output["metrics"],
        "skill": output["skill_used"],
    }


def _run_skill_only(
    message: str,
    skill_name: str,
    model: str,
    temperature: float,
    use_api: bool,
    orchestrator: Any,
) -> dict[str, Any]:
    """Run a skill from chat text without sending the result to the model."""
    if use_api:
        skill = _skill_by_name(skill_name, True, None)
        payload = _map_message(skill_name, message) if skill is None else _map_from_schema(skill, message)
        try:
            response = requests.post(
                f"{API_BASE}/api/skills/execute",
                json={
                    "skill_name": skill_name,
                    "input_data": payload,
                    "use_llm": False,
                    "model": model,
                    "temperature": temperature,
                },
                timeout=60,
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text) if response.content else response.text
                return {
                    "content": f"Request failed ({response.status_code}): {detail}",
                    "metrics": {},
                    "skill": skill_name,
                }
            body = response.json()
            return {
                "content": _render_result(body.get("result")),
                "metrics": body.get("metrics") or {},
                "skill": skill_name,
            }
        except requests.RequestException as exc:
            return {"content": f"API request failed: {exc}", "metrics": {}, "skill": skill_name}

    skill = next(item for item in orchestrator.skills.values() if item.name == skill_name)
    payload = orchestrator.map_message_to_input(skill, message)
    output = orchestrator.execute_skill(skill_name, payload, False, None, model, temperature)
    return {
        "content": _render_result(output["result"]),
        "metrics": output["metrics"],
        "skill": skill_name,
    }


def _skill_by_name(name: str, use_api: bool, orchestrator: Any) -> dict[str, Any] | None:
    """Find one registry entry by name."""
    for entry in fetch_skills(use_api, orchestrator):
        if entry.get("name") == name:
            return entry
    return None


def _map_message(skill_name: str, message: str) -> dict[str, Any]:
    """Fallback chat mapping used when the registry entry is unavailable."""
    if skill_name == "github_pr_summarizer":
        return {"repo": "octocat/Hello-World", "pr_number": 1}
    if skill_name == "code_reviewer":
        return {"code": message}
    if skill_name == "rag_doc_search":
        return {"query": message}
    return {"text": message}


def _map_from_schema(skill: dict[str, Any], message: str) -> dict[str, Any]:
    """Map chat text onto the first required string field of a registry entry."""
    if skill.get("name") == "github_pr_summarizer":
        import re

        match = re.search(r"([\w.-]+/[\w.-]+)\s*#?\s*(\d+)", message)
        if match:
            return {"repo": match.group(1), "pr_number": int(match.group(2))}
        return {"repo": "octocat/Hello-World", "pr_number": 1}
    schema = skill.get("input_schema") or {}
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    for key in required:
        prop = properties.get(key) or {}
        if prop.get("type") == "string":
            return {key: message}
        if prop.get("type") == "integer":
            return {key: 1}
    return {required[0]: message} if required else {"text": message}


def _render_result(result: Any) -> str:
    """Render a skill result with the same rules as the harness."""
    from backend.orchestrator import AgentOrchestrator

    return AgentOrchestrator._render_result(result)


def execute_payload(
    skill_name: str,
    payload: dict[str, Any],
    use_llm: bool,
    model: str,
    temperature: float,
    use_api: bool,
    orchestrator: Any,
) -> dict[str, Any]:
    """Run a skill from the Skill Hub form."""
    if use_api:
        try:
            response = requests.post(
                f"{API_BASE}/api/skills/execute",
                json={
                    "skill_name": skill_name,
                    "input_data": payload,
                    "use_llm": use_llm,
                    "model": model,
                    "temperature": temperature,
                },
                timeout=180,
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text) if response.content else response.text
                return {"error": f"{response.status_code}: {detail}"}
            return response.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}
    try:
        return orchestrator.execute_skill(skill_name, payload, use_llm, None, model, temperature)
    except Exception as exc:
        return {"error": str(exc)}


def chat_markdown(messages: list[dict[str, Any]]) -> str:
    """Render the session transcript as Markdown."""
    lines = ["# AgentFlow chat export", ""]
    if not messages:
        lines.append("_No messages yet._")
        return "\n".join(lines)
    for message in messages:
        role = str(message.get("role", "assistant")).title()
        lines.append(f"## {role}")
        if message.get("skill"):
            lines.append(f"_Skill: {message['skill']}_")
        lines.append("")
        lines.append(str(message.get("content", "")))
        lines.append("")
    return "\n".join(lines)


def save_exports(messages: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write the current transcript to ``exports/`` as JSON and Markdown."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = EXPORTS_DIR / f"chat-{stamp}.json"
    md_path = EXPORTS_DIR / f"chat-{stamp}.md"
    json_path.write_text(json.dumps(messages, indent=2, default=str), encoding="utf-8")
    md_path.write_text(chat_markdown(messages), encoding="utf-8")
    return json_path, md_path


def schema_widget(skill_name: str, key: str, prop: dict[str, Any]) -> Any:
    """Build one Streamlit input from a JSON Schema property."""
    description = str(prop.get("description") or "")
    json_type = prop.get("type", "string")
    widget_key = f"{skill_name}-{key}"
    if json_type == "integer":
        default = 3 if key == "max_sentences" else 1 if key == "pr_number" else 3 if key == "top_k" else 1
        return int(st.number_input(key, min_value=1, value=default, help=description, key=widget_key))
    if json_type == "number":
        return float(st.number_input(key, value=0.0, help=description, key=widget_key))
    if json_type == "boolean":
        return bool(st.checkbox(key, help=description, key=widget_key))
    if key == "github_token":
        return st.text_input(key, type="password", help=description, key=widget_key)
    if key in {"text", "code", "query"}:
        return st.text_area(key, value=_sample_text(key), height=140, help=description, key=widget_key)
    default_text = "octocat/Hello-World" if key == "repo" else "python" if key == "language" else ""
    if key == "data_dir":
        default_text = "data/sample_notes"
    return st.text_input(key, value=default_text, help=description, key=widget_key)


def _sample_text(key: str) -> str:
    """Return a useful default for the Skill Hub test form."""
    if key == "code":
        return "from os import *\n\ndef run(items=[]):\n    try:\n        return items\n    except:\n        return None\n"
    if key == "query":
        return "How does chunking reduce hallucination during retrieval?"
    return (
        "Agents plan a step, call a tool, and read the result before they continue. "
        "A harness checks that the tool really ran. Local models stay useful when that check is automatic."
    )


def render_skill_card(skill: dict[str, Any]) -> None:
    """Draw one registry card with the compliance badge and schema."""
    name = html.escape(str(skill.get("name", "")))
    description = html.escape(str(skill.get("description", "")))
    version = html.escape(str(skill.get("version", "")))
    st.markdown(
        f"""
        <div class="af-card">
            <div class="af-title"><code>{name}</code></div>
            <div class="af-desc">{description}</div>
            <span class="af-badge">Agent Skill Open Standard ✅</span>
            <span class="af-version">v{version}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(json.dumps(skill.get("input_schema") or {}, indent=2), language="json")


def average_latency(history: list[dict[str, Any]]) -> float:
    """Average ``execution_time_ms`` across the visible history."""
    if not history:
        return 0.0
    total = sum(float(item.get("execution_time_ms") or 0) for item in history)
    return total / len(history)


def top_skill(history: list[dict[str, Any]]) -> str:
    """Return the skill name that appears most often in history."""
    counts: dict[str, int] = {}
    for item in history:
        name = str(item.get("skill_name") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "—"
    return max(counts, key=lambda name: counts[name])


_init_state()
use_api = api_reachable(API_BASE)
st.session_state["mode"] = "api" if use_api else "embedded"
orchestrator: Any = None

with st.sidebar:
    st.header("AgentFlow")
    if use_api:
        st.markdown('<div class="af-mode-api">Mode: FastAPI</div>', unsafe_allow_html=True)
        st.caption(API_BASE)
    else:
        st.markdown('<div class="af-mode-local">Mode: embedded harness</div>', unsafe_allow_html=True)
        st.caption("FastAPI is not reachable, so skills run inside this app.")

    ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA)
    if st.button("Test Connection"):
        ok, names, message = probe_ollama(ollama_url)
        st.session_state["ollama_probe"] = {"ok": ok, "names": names, "message": message}
    probe = st.session_state.get("ollama_probe")
    if probe and probe["ok"]:
        st.success("✅ " + probe["message"])
        if probe["names"]:
            st.caption(", ".join(probe["names"]))
    elif os.getenv("RENDER"):
        st.info(OLLAMA_OFFLINE)
    elif probe:
        st.warning(probe["message"])

    model = st.selectbox("Model", MODELS, index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    if not use_api:
        orchestrator = local_orchestrator(ollama_url)

    st.divider()
    st.subheader("Export chat")
    transcript = st.session_state["messages"]
    st.download_button(
        "Download JSON",
        data=json.dumps(transcript, indent=2, default=str),
        file_name="agentflow-chat.json",
        mime="application/json",
    )
    st.download_button(
        "Download Markdown",
        data=chat_markdown(transcript),
        file_name="agentflow-chat.md",
        mime="text/markdown",
    )
    if st.button("Save to exports/"):
        json_path, md_path = save_exports(transcript)
        st.success(f"Saved {json_path.name} and {md_path.name}")

    st.divider()
    st.subheader("System")
    status = fetch_status(use_api, orchestrator)
    if status.get("error"):
        st.warning(status["error"])
    else:
        connected = "yes" if status.get("ollama_connected") else "no"
        st.write(f"Ollama connected: {connected}")
        st.write(f"Skills loaded: {len(status.get('loaded_skills') or [])}")
        st.write(f"Requests: {status.get('total_requests', 0)}")
        st.write(f"Uptime (s): {status.get('uptime_seconds', 0)}")

workspace, registry, metrics_tab = st.tabs(
    ["Agent Workspace", "Skill Hub", "System Metrics & Logs"]
)

with workspace:
    st.subheader("Agent workspace")
    st.caption("Pick a skill, or stay on Direct Chat and talk to the model harness.")
    skill_entries = fetch_skills(use_api, orchestrator)
    skill_names = ["Direct Chat"] + [str(item["name"]) for item in skill_entries]
    selected = st.selectbox("Skill", skill_names)
    enhance = st.checkbox("Enhance with LLM", value=True)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message.get("skill"):
                st.caption(f"Skill: {message['skill']}")
            st.markdown(message["content"])
            if message.get("metrics"):
                with st.expander("Execution metrics"):
                    st.json(message["metrics"])
    prompt = st.chat_input("Ask AgentFlow, paste notes, or paste code")
    if prompt:
        skill_name = None if selected == "Direct Chat" else selected
        st.session_state["messages"].append({"role": "user", "content": prompt, "skill": skill_name})
        with st.spinner("Running the harness..."):
            try:
                turn = run_turn(
                    prompt,
                    skill_name,
                    enhance,
                    model,
                    float(temperature),
                    use_api,
                    orchestrator,
                )
            except Exception as exc:
                turn = {
                    "content": f"The harness could not complete that turn: {exc}",
                    "metrics": {},
                    "skill": skill_name,
                }
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": turn["content"],
                "metrics": turn["metrics"],
                "skill": turn["skill"],
            }
        )
        st.rerun()

with registry:
    st.subheader("Skill hub")
    st.caption("Every card is an Agent Skill Open Standard v1.0 registry entry.")
    if not skill_entries:
        st.info("No skills are loaded.")
    for skill in skill_entries:
        render_skill_card(skill)
        with st.expander(f"Test {skill['name']}"):
            payload: dict[str, Any] = {}
            properties = (skill.get("input_schema") or {}).get("properties") or {}
            for key, prop in properties.items():
                if isinstance(prop, dict):
                    payload[key] = schema_widget(str(skill["name"]), key, prop)
            run_llm = st.checkbox(
                "Enhance this run with the LLM",
                value=False,
                key=f"llm-{skill['name']}",
            )
            if st.button("Run", key=f"run-{skill['name']}"):
                cleaned = {
                    key: value
                    for key, value in payload.items()
                    if value not in ("", None)
                }
                with st.spinner(f"Running {skill['name']}..."):
                    result = execute_payload(
                        str(skill["name"]),
                        cleaned,
                        run_llm,
                        model,
                        float(temperature),
                        use_api,
                        orchestrator,
                    )
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.markdown(_render_result(result.get("result")))
                    if result.get("llm_response"):
                        st.markdown(f"**LLM insight** ({result.get('llm_source')})")
                        st.markdown(result["llm_response"])
                    with st.expander("Raw response"):
                        st.json(result)

with metrics_tab:
    st.subheader("System metrics")
    history = fetch_history(use_api, orchestrator)
    status = fetch_status(use_api, orchestrator)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total requests", int(status.get("total_requests") or 0))
    col_b.metric("Average latency (ms)", round(average_latency(history), 1))
    col_c.metric("Top skill", top_skill(history))
    if history:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No executions yet. Run a skill or send a chat message.")
    connected = "connected" if status.get("ollama_connected") else "not connected"
    models = ", ".join(status.get("ollama_models") or []) or "none reported"
    st.write(f"Ollama is {connected}. Models: {models}.")
    st.write(f"Uptime: {status.get('uptime_seconds', 0)} seconds.")
    if st.button("Refresh"):
        st.rerun()
