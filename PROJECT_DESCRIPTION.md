# Project Description

## Mission
Lang Teacher Agent is an AI tutor focused on helping learners practice Greek through conversational interactions. The agent provides contextual answers powered by OpenAI's Responses API and retains a short history of each chat so learners can iterate on their questions.

## System Architecture
- `src/main.py`: Thin entry point that loads environment-backed settings and delegates to the runtime bootstrap.
- `src/app/`: Separates environment configuration (`settings.py`) from runtime wiring (`runtime.py`), including logging setup and migration execution.
- `src/bot/`: Contains the `GreekTeacherAgent` Telegram handler, flashcard extraction workflow (`flashcard_workflow.py`), spaced-repetition scheduling helpers (`srs.py`), supporting components in `agent_components/` (image helpers live in `image_support.py`), and the Telegram application factory.
- `src/services/`: Wraps external integrations, currently limited to the OpenAI client builder.
- `src/db/`: Houses SQLAlchemy models, flashcard persistence helpers, session management, and helpers for running migrations during startup.
- Telegram Bot API: Receives messages from learners and relays them to the OpenAI-backed tutor, which responds with tailored Greek language guidance.
- Flashcard flow: User intent is detected in free-form messages, OpenAI synthesises translations/examples, and cards are stored in shared (`flashcards`) and user-specific (`user_flashcards`, `flashcard_reviews`) tables. Inline Telegram buttons present "Взять карточку" to surface a partially hidden card, "Показать полностью" to reveal the rest, and 1-5 rating buttons that feed the SM-2 scheduler. When learners send a generic follow-up such as “добавь в карточки”, the workflow keeps only the primary translation from the latest explanation; extra synonyms or grammatical variants are saved only if the message explicitly lists the Greek terms or asks for all variants.
- Alembic migrations: Applied automatically when the application boots so the database schema stays up to date.
- PostgreSQL (via Docker): Stores Telegram user records alongside shared flashcards, per-user scheduling metadata, and review history.
- GitHub Actions workflow: Runs tests and Docker builds to keep the main branch production-ready.

## Directory Structure
- `src/`: Application source code organised into focused packages.
- `src/app/`: Configuration loading and runtime orchestration.
- `src/bot/`: Telegram-facing agent logic, flashcard workflow, application wiring, and extracted helper modules such as `agent_components/image_support.py` for image processing.
- `src/db/`: Database models, session factory, flashcard helpers, and migration utilities.
- `src/services/`: External service helpers (OpenAI client).
- `tests/`: Pytest-based async test suite covering conversation memory, flashcard storage, workflow parsing, and SRS scheduling.
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
- Follow the development rules in `AGENTS.md` to keep the codebase token-efficient, protect Russian localisation fidelity, and uphold professional user communication.
Changes are not complete until both documents have been reviewed and amended accordingly.
