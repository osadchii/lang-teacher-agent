# Project Description

## Mission
Lang Teacher Agent is an AI tutor focused on helping learners practice Greek through conversational interactions. The agent provides contextual answers powered by OpenAI's Responses API and retains a short history of each chat so learners can iterate on their questions.

## System Architecture
- `src/main.py`: Boots the application, configures logging, starts the Telegram bot, and maintains the short-term memory of each chat.
- Telegram Bot API: Receives messages from learners and relays them to the OpenAI-backed tutor, which responds with tailored Greek language guidance.
- PostgreSQL (via Docker): Reserved for persisting learner progress and lesson content in future milestones.
- GitHub Actions workflow: Runs tests and Docker builds to keep the main branch production-ready.

## Directory Structure
- `src/`: Application source code. The root module exposes the Telegram bot entry point.
- `tests/`: Pytest-based test suite (placeholder for future test coverage).
- `.github/workflows/`: Continuous integration workflows.
- `docker-compose.yml`: Local development stack wiring the app and PostgreSQL.
- `Dockerfile`: Container image definition for deployment.
- `.env.example`: Template with all environment variables required to run the application.

## Update Policy
Every time the project changes, maintainers must:
- Update `README.md` so onboarding instructions stay accurate.
- Update `PROJECT_DESCRIPTION.md` so this document reflects the latest architecture and structure.
Changes are not complete until both documents have been reviewed and amended accordingly.
