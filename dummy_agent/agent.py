"""Moved to `agenteval.demo.banking_agent`. Run it with `agenteval demo-agent`.

Kept so `uvicorn agent:app` from this folder still works (needs `pip install agenteval`).
"""

from agenteval.demo.banking_agent import app  # noqa: F401
