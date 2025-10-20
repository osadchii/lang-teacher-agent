import asyncio
import logging
import os
from collections import deque
from typing import Deque, Dict, Final, List, Optional, Tuple

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

try:
    from src.db import get_session_factory, run_migrations_if_needed
    from src.db.users import upsert_user
except ModuleNotFoundError as exc:
    if exc.name not in {"src", "src.db", "src.db.users"}:
        raise
    from db import get_session_factory, run_migrations_if_needed
    from db.users import upsert_user


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

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        history_size: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._history_size = history_size
        self._history: Dict[int, Deque[Tuple[str, str]]] = {}
        self._session_factory = session_factory

    def _get_history(self, chat_id: int) -> Deque[Tuple[str, str]]:
        """Return the rolling history buffer for a chat, creating it when needed."""
        history = self._history.get(chat_id)
        if history is None:
            history = deque(maxlen=self._history_size)
            self._history[chat_id] = history
        return history

    def _record_interaction(self, chat_id: int, user_message: str, assistant_reply: str) -> None:
        """Persist the most recent user/assistant exchange for the chat."""
        history = self._get_history(chat_id)
        history.append((user_message, assistant_reply))

    async def _store_user_profile(
        self,
        chat_id: int,
        first_name: Optional[str],
        last_name: Optional[str],
    ) -> None:
        """Save the Telegram user's profile details into the database."""
        if self._session_factory is None:
            return

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await upsert_user(session, chat_id, first_name, last_name)
        except Exception:
            LOGGER.exception("Failed to upsert Telegram user record for chat %s.", chat_id)

    def _build_messages(self, chat_id: int, user_message: str) -> List[Dict[str, str]]:
        """Compose the conversation context to send to the OpenAI Responses API."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        for previous_user_message, previous_reply in self._get_history(chat_id):
            messages.append({"role": "user", "content": previous_user_message})
            messages.append({"role": "assistant", "content": previous_reply})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _generate_response(self, chat_id: int, user_message: str) -> str:
        """Call the OpenAI Responses API and return plain text."""
        response = await self._client.responses.create(
            model=self._model,
            input=self._build_messages(chat_id, user_message),
        )
        return extract_output_text(response)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process an incoming Telegram message and reply with the AI-generated answer."""
        if not update.message or not update.message.text:
            return

        chat = update.effective_chat
        if chat is None:
            return

        user_message = update.message.text.strip()
        if not user_message:
            return

        user = update.effective_user
        if user is not None:
            await self._store_user_profile(chat.id, getattr(user, "first_name", None), getattr(user, "last_name", None))

        try:
            reply = await self._generate_response(chat.id, user_message)
        except Exception:
            LOGGER.exception("Failed to generate response from OpenAI.")
            reply = (
                "Failed to reach the Greek tutor right now. Please try your request again in a moment."
            )

        if reply:
            await update.message.reply_text(reply)
            self._record_interaction(chat.id, user_message, reply)


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

    try:
        run_migrations_if_needed()
    except Exception:
        LOGGER.exception("Database migrations failed. Aborting startup.")
        raise

    session_factory = get_session_factory()
    openai_client = build_openai_client(api_key)
    agent = GreekTeacherAgent(openai_client, model_name, SYSTEM_PROMPT, session_factory=session_factory)
    application = create_application(bot_token, agent)

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    LOGGER.info("Starting Telegram bot for %s in %s mode.", app_name, app_env)
    application.run_polling()


if __name__ == "__main__":
    main()
