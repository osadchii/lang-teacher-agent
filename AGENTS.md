# Codex Collaboration Rules

1. All communication, documentation, and code comments must be in English.
2. Every change to the project must update both `README.md` and `PROJECT_DESCRIPTION.md` before it is considered complete.
3. Keep configuration samples (such as `.env.example`) aligned with the settings required to run the application and update them whenever new variables are introduced.
4. When altering the database schema, generate an Alembic migration, review it into source control, and document any new workflow steps for contributors.
5. Ensure the codebase remains well structured into clear packages/modules and that the application stays runnable both locally and via Docker.

## Development Rules

### Before You Start
- Read `README.md`, `PROJECT_DESCRIPTION.md`, and recent commit history to understand context and avoid duplicating work.
- Clarify scope early: capture the user goal in a short plan, call out assumptions, and confirm any risky decisions before coding.
- Identify affected modules up front and prefer touching the smallest possible surface area.

### Implementation Guidelines
- Follow the existing package boundaries (`src/app`, `src/bot`, `src/db`, `src/services`) and keep new logic in the module that owns the responsibility.
- Keep functions cohesive and small; extract reusable helpers instead of copying prompts or workflows.
- Maintain type hints and docstrings where they already exist; adopt the project’s logging and error-handling patterns when extending features.
- When introducing configuration, add defaults in `src/app/settings.py`, document them, and sync `.env` / `.env.example`.

### Token-Efficient Codebase
- Keep source files and prompt templates focused; extract modules when files grow past a single responsibility to avoid ballooning context windows.
- Prefer well-scoped helpers and shared prompt builders instead of duplicating near-identical strings or workflows.
- Document public functions and complex prompts with concise docstrings or comments that explain intent, inputs, and failure modes.
- Remove obsolete code paths, unused imports, and dead prompts so only live tokens remain in the repository and runtime context.
- When editing localisation assets, verify Russian text survives encoding changes (UTF-8) and displays correctly in prompts, tests, and UI strings.

### Quality & Handover
- Run the relevant automated checks (`pytest`, lint, type checks, Docker build) that cover the changed area and note any gaps in the final message.
- Update `README.md`, `PROJECT_DESCRIPTION.md`, and task-specific docs with new behaviours, settings, or workflows.
- Record edge cases, rollback steps, and follow-up ideas so the next contributor can resume quickly.
- Keep commits focused and reference migrations, schema updates, and environment changes explicitly in their messages.

### User Communication
- Provide comprehensive, professional responses to users; do not truncate answers solely to save tokens.
- Surface risks, assumptions, and validation status clearly so users can make informed decisions.
