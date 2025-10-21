# Project Description

## Mission
Lang Teacher Agent is an AI tutor focused on helping learners practice Greek through conversational interactions. The agent provides contextual answers powered by OpenAI's Responses API and retains a short history of each chat so learners can iterate on their questions.

## System Architecture
- `src/main.py`: Thin entry point that loads environment-backed settings and delegates to the runtime bootstrap.
- `src/app/`: Separates environment configuration (`settings.py`) from runtime wiring (`runtime.py`), including logging setup and migration execution.
- `src/bot/`: Contains the `GreekTeacherAgent` Telegram handler and the Telegram application factory.
- `src/services/`: Wraps external integrations, currently limited to the OpenAI client builder.
- `src/db/`: Houses SQLAlchemy models, session management, and helpers for running migrations during startup.
- Telegram Bot API: Receives messages from learners and relays them to the OpenAI-backed tutor, which responds with tailored Greek language guidance.
- Alembic migrations: Applied automatically when the application boots so the database schema stays up to date.
- PostgreSQL (via Docker): Stores Telegram user records with their identifiers, names, and first-contact timestamps.
- GitHub Actions workflow: Runs tests and Docker builds to keep the main branch production-ready.

## Directory Structure
- `src/`: Application source code organised into focused packages.
- `src/app/`: Configuration loading and runtime orchestration.
- `src/bot/`: Telegram-facing agent logic and application wiring.
- `src/db/`: Database models, session factory, and migration utilities.
- `src/services/`: External service helpers (OpenAI client).
- `tests/`: Pytest-based test suite (placeholder for future test coverage).
- `migrations/`: Alembic migration scripts tracked by `alembic.ini`.
- `.github/workflows/`: Continuous integration workflows.
- `docker-compose.yml`: Local development stack wiring the app and PostgreSQL.
- `Dockerfile`: Container image definition for deployment.
- `.env.example`: Template with all environment variables required to run the application.
- `alembic.ini`: Alembic configuration used by the automated migration runner.

## Update Policy
Every time the project changes, maintainers must:
- Update `README.md` so onboarding instructions stay accurate.
- Update `PROJECT_DESCRIPTION.md` so this document reflects the latest architecture and structure.
Changes are not complete until both documents have been reviewed and amended accordingly.
