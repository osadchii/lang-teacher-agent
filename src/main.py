import logging
import os
from typing import Final

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
LOGGER: Final = logging.getLogger(__name__)
DEFAULT_MODEL: Final = "gpt-5-mini"
SYSTEM_PROMPT: Final = (
    "You are an expert Greek language teacher. Help learners translate words and phrases, explain grammar "
    "rules when asked, and provide example sentences and pronunciation guidance. Whenever you supply a Greek "
    "noun, include the correct definite article so the gender is clear."
)


def extract_output_text(response: object) -> str:
    """Best-effort extraction of text from an OpenAI Responses result."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        collected = []
        for item in output:
            maybe_text = getattr(getattr(item, "content", None), "text", None)
            if isinstance(maybe_text, list):
                for part in maybe_text:
                    if getattr(part, "type", None) == "output_text":
                        value = getattr(part, "value", None)
                        if value:
                            collected.append(str(value))
            elif isinstance(maybe_text, str):
                collected.append(maybe_text)
        if collected:
            return "\n".join(collected)

    return ""


class GreekTeacherAgent:
    """Handles Telegram messages by delegating to an OpenAI Greek teacher model."""

    def __init__(self, client: AsyncOpenAI, model: str, system_prompt: str) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt

    async def _generate_response(self, user_message: str) -> str:
        """Call the OpenAI Responses API and return plain text."""
        response = await self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return extract_output_text(response)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process an incoming Telegram message and reply with the AI-generated answer."""
        if not update.message or not update.message.text:
            return

        user_message = update.message.text.strip()
        if not user_message:
            return

        try:
            reply = await self._generate_response(user_message)
        except Exception:
            LOGGER.exception("Failed to generate response from OpenAI.")
            reply = (
                "Failed to reach the Greek tutor right now. Please try your request again in a moment."
            )

        if reply:
            await update.message.reply_text(reply)


def build_openai_client(api_key: str) -> AsyncOpenAI:
    """Create a configured AsyncOpenAI client."""
    return AsyncOpenAI(api_key=api_key)


def create_application(bot_token: str, agent: GreekTeacherAgent) -> Application:
    """Configure the Telegram application instance."""
    application = ApplicationBuilder().token(bot_token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, agent.handle_message))
    return application


def main() -> None:
    """Entry point for the application."""
    app_name = os.getenv("APP_NAME", "Lang Teacher Agent")
    app_env = os.getenv("APP_ENV", "development")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    print(f"{app_name} is running in {app_env} mode.")

    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required to start the Telegram bot.")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to generate answers.")

    openai_client = build_openai_client(api_key)
    agent = GreekTeacherAgent(openai_client, model_name, SYSTEM_PROMPT)
    application = create_application(bot_token, agent)

    LOGGER.info("Starting Telegram bot for %s in %s mode.", app_name, app_env)
    application.run_polling()


if __name__ == "__main__":
    main()
