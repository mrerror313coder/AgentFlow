"""AgentFlow backend package.

The orchestrator in this package is the model harness: it discovers skills,
runs them, and optionally asks an open-weight model served by Ollama to
interpret the result.
"""

__all__ = ["models"]
