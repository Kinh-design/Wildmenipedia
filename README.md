# Wildmenipedia (GrokAlternative) – Agentic Knowledge System Scaffold

This repository contains a minimal, production‑ready scaffold for an agentic, world‑scale knowledge system:

- FastAPI service (API)
- LangGraph/LangChain‑ready structure for agents
- Neo4j (graph) + Qdrant (vector) via Docker Compose
- CI with linting, type checks, and tests

## Quickstart

1) Create and fill `.env` from `.env.example`.

2) Install with Poetry:

```bash
pip install --upgrade pip
pip install poetry
poetry install
poetry run pytest -q
```

3) Run API:

```bash
poetry run uvicorn grokalternative.api:app --reload
```

4) Optional: start services via Docker Compose:

```bash
docker compose up -d
```
