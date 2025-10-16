import logging
import os
from typing import Final

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
LOGGER: Final = logging.getLogger(__name__)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply with the same text the user sent."""
    if not update.message or not update.message.text:
        return

    await update.message.reply_text(update.message.text)


def create_application(bot_token: str) -> Application:
    """Configure the Telegram application instance."""
    application = ApplicationBuilder().token(bot_token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    return application


def main() -> None:
    """Entry point for the application."""
    app_name = os.getenv("APP_NAME", "Lang Teacher Agent")
    app_env = os.getenv("APP_ENV", "development")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    print(f"{app_name} is running in {app_env} mode.")

    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required to start the Telegram bot.")

    application = create_application(bot_token)
    LOGGER.info("Starting Telegram bot for %s in %s mode.", app_name, app_env)
    application.run_polling()


if __name__ == "__main__":
    main()
