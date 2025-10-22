"""Telegram application wiring for the Lang Teacher Agent."""

from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from .agent import GreekTeacherAgent


def build_application(bot_token: str, agent: GreekTeacherAgent) -> Application:
    """Configure the Telegram application instance."""
    application = ApplicationBuilder().token(bot_token).build()
    application.add_handler(CommandHandler("start", agent.handle_start))
    application.add_handler(CallbackQueryHandler(agent.handle_take_flashcard, pattern="^fc_take$"))
    application.add_handler(CallbackQueryHandler(agent.handle_delete_flashcard, pattern=r"^fc_delete:"))
    application.add_handler(CallbackQueryHandler(agent.handle_show_flashcard, pattern=r"^fc_show:"))
    application.add_handler(CallbackQueryHandler(agent.handle_rate_flashcard, pattern=r"^fc_rate:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, agent.handle_message))
    return application
