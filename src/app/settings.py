"""Configuration helpers for the Lang Teacher Agent runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_HISTORY_SIZE = 5
SYSTEM_PROMPT = (
    "You are an expert Greek language teacher. Help learners translate words and phrases, explain grammar "
    "rules when asked, and provide example sentences and pronunciation guidance. Whenever you supply a Greek "
    "noun, include the correct definite article so the gender is clear. Keep answers short and to the point. "
    "Do not add examples or long explanations unless the learner explicitly asks for them. You may use "
    "Telegram-friendly formatting (bold, italics, lists) when it improves clarity."
)


@dataclass(frozen=True)
class AppSettings:
    """Strongly typed application settings loaded from environment variables."""

    app_name: str
    app_env: str
    log_level: str
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    history_size: int

    @classmethod
    def from_env(cls) -> AppSettings:
        """Construct settings directly from environment variables."""
        app_name = os.getenv("APP_NAME", "Lang Teacher Agent")
        app_env = os.getenv("APP_ENV", "development")
        log_level = os.getenv("LOG_LEVEL", "INFO")
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

        if not telegram_bot_token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN environment variable is required to start the Telegram bot."
            )

        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required to generate answers.")

        try:
            history_size = int(os.getenv("TEACHER_HISTORY_SIZE", str(DEFAULT_HISTORY_SIZE)))
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise RuntimeError("TEACHER_HISTORY_SIZE must be an integer.") from exc

        if history_size < 1:
            raise RuntimeError("TEACHER_HISTORY_SIZE must be a positive integer.")

        return cls(
            app_name=app_name,
            app_env=app_env,
            log_level=log_level,
            telegram_bot_token=telegram_bot_token,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            history_size=history_size,
        )
