# Lang Teacher Agent

Lang Teacher Agent is an AI-powered tutor that helps learners practise the Greek language. The application exposes its functionality through a Telegram bot and is built with Python, Docker, and PostgreSQL.

## Overview
- Telegram bot that connects to OpenAI's Responses API to deliver explanations, translations, and pronunciation guidance.
- Remembers the last five user messages and assistant replies per chat so learners can ask follow-up questions without repeating context (configurable through `TEACHER_HISTORY_SIZE`).
- Persists Telegram user identifiers and profile details in PostgreSQL, capturing the first time a learner contacts the bot.
- Configurable through environment variables stored in `.env` or provided at runtime.
- Docker Compose environment that bundles the application and PostgreSQL for local development.
- Continuous integration workflow defined in GitHub Actions.
- Structured Python packages (`src/app`, `src/bot`, `src/db`, `src/services`) keep configuration, runtime wiring, and external integrations isolated.

## Prerequisites
- Python 3.11+
- pip (examples below use pip)
- Docker and Docker Compose (optional, for containerized runs)

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` (or your shell environment) with valid `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`, and database settings (`DATABASE_URL`, `POSTGRES_*`). The flag `RUN_MIGRATIONS_ON_STARTUP` controls whether migrations are applied automatically when the process boots (defaults to `true`). The short-term memory buffer size can be tuned with `TEACHER_HISTORY_SIZE`.

Start the application:

```bash
python -m src.main
```

## Database & Migrations
- Alembic migrations live in the `migrations/` directory and are configured via `alembic.ini`.
- On startup the bot calls `alembic upgrade head` (through `src.db.run_migrations_if_needed`) so the schema is always up to date.
- To create a new migration, run `alembic revision --autogenerate -m "describe change"` with the virtual environment activated, review the generated script, and commit it alongside the relevant model changes.

## Testing
```bash
pytest
```

## VS Code
Open the folder in VS Code to pick up `.vscode/launch.json`. Use the **Lang Teacher Agent (local)** configuration to debug `src.main` with environment variables from `.env`. To run the Docker stack from VS Code, trigger the `docker-compose: up` task (`Terminal → Run Task…`).

## Docker Usage
Build and run the services (application + PostgreSQL) with:
```bash
docker compose up --build
```

## Continuous Integration
Automated checks run in `.github/workflows/ci.yml`. The workflow installs dependencies, runs the test suite, and builds the Docker image to ensure each change keeps the project healthy.

## Additional Documentation
- `PROJECT_DESCRIPTION.md` records the current architecture and directory structure.
- `CODEX_RULES.md` captures collaboration rules for future Codex sessions.
## Project Structure
- `src/app/`: Application configuration (`settings.py`) and runtime bootstrap (`runtime.py`).
- `src/bot/`: Telegram agent, message handlers, and wiring to the Telegram SDK.
- `src/db/`: SQLAlchemy models, migration utilities, and persistence helpers.
- `src/services/`: Integrations with third-party providers such as OpenAI.
- `tests/`: Pytest-based test suite.
- Remaining top-level files configure Docker, Alembic, and CI.
