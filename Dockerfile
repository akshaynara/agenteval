FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY frontend ./frontend
RUN pip install --no-cache-dir ".[google]" && useradd --create-home agenteval
USER agenteval
WORKDIR /work

# Web UI by default:   docker run -p 8000:8000 agenteval
# CLI:                 docker run -v "$PWD:/work" -e ANTHROPIC_API_KEY agenteval run
EXPOSE 8000
ENTRYPOINT ["agenteval"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
