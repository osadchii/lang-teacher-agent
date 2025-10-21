"""Telegram application wiring for the Lang Teacher Agent."""

from telegram.ext import Application, ApplicationBuilder, MessageHandler, filters

from .agent import GreekTeacherAgent


def build_application(bot_token: str, agent: GreekTeacherAgent) -> Application:
    """Configure the Telegram application instance."""
    application = ApplicationBuilder().token(bot_token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, agent.handle_message))
    return application
