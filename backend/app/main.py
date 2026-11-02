"""Compatibility entry point for the existing Render deploy (`uvicorn app.main:app`, root dir `backend`).

The server now lives in the `agenteval` package: `agenteval serve`, or `uvicorn agenteval.server:app`.
"""

from agenteval.server import app  # noqa: F401
