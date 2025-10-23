"""Bootstrap logic for running the Telegram bot."""

from __future__ import annotations

import asyncio
import logging

from src.app.settings import AppSettings, SYSTEM_PROMPT
from src.bot import GreekTeacherAgent, build_application
from src.db import get_session_factory, run_migrations_if_needed
from src.services import build_openai_client


LOGGER = logging.getLogger(__name__)


def _configure_logging(log_level: str) -> None:
    """Set up project-wide logging configuration."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=log_level,
    )


def _ensure_event_loop() -> None:
    """Guarantee that an asyncio event loop exists for the current thread."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def run_bot(settings: AppSettings) -> None:
    """Start the Telegram bot using the provided settings."""
    _configure_logging(settings.log_level)
    print(f"{settings.app_name} is running in {settings.app_env} mode.")

    try:
        run_migrations_if_needed()
    except Exception:
        LOGGER.exception("Database migrations failed. Aborting startup.")
        raise

    session_factory = get_session_factory()
    openai_client = build_openai_client(settings.openai_api_key)
    agent = GreekTeacherAgent(
        openai_client,
        settings.openai_model,
        SYSTEM_PROMPT,
        session_factory=session_factory,
        history_size=settings.history_size,
        flashcard_model=settings.flashcard_model,
        flashcard_source_language=settings.flashcard_source_language,
        flashcard_target_language=settings.flashcard_target_language,
        flashcard_max_cards=settings.flashcard_max_cards,
        vision_model=settings.openai_vision_model,
    )
    application = build_application(settings.telegram_bot_token, agent)

    _ensure_event_loop()

    LOGGER.info("Starting Telegram bot for %s in %s mode.", settings.app_name, settings.app_env)
    application.run_polling()
