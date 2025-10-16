# Project Description

## Mission
Lang Teacher Agent is an AI tutor focused on helping learners practice Greek through conversational interactions. The initial milestone is an echo-style Telegram bot that will gradually evolve into a guided language learning experience.

## System Architecture
- `src/main.py`: Boots the application, configures logging, and starts the Telegram bot.
- Telegram Bot API: Receives messages from learners and currently echoes them back.
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
