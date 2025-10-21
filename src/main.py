from src.app import AppSettings, run_bot
from src.app.settings import SYSTEM_PROMPT
from src.bot.agent import GreekTeacherAgent

__all__ = ["main", "GreekTeacherAgent", "SYSTEM_PROMPT"]


def main() -> None:
    """Entry point for the application."""
    settings = AppSettings.from_env()
    run_bot(settings)


if __name__ == "__main__":
    main()
