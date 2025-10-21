"""Telegram handlers for the Greek language teacher agent."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from contextlib import suppress
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from src.bot.flashcard_workflow import FlashcardWorkflow, FlashcardWorkflowResult
from src.bot.openai_utils import extract_output_text
from src.bot.srs import calculate_next_schedule
from src.db import UserFlashcard
from src.db.flashcards import get_next_flashcard_for_user, record_flashcard_review
from src.db.users import upsert_user


LOGGER = logging.getLogger(__name__)


class GreekTeacherAgent:
    """Handles Telegram messages by delegating to an OpenAI Greek teacher model."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        history_size: int = 5,
        flashcard_model: Optional[str] = None,
        flashcard_source_language: str = "Greek",
        flashcard_target_language: str = "Russian",
        flashcard_max_cards: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._history_size = history_size
        self._history: Dict[int, Deque[Tuple[str, str]]] = {}
        self._session_factory = session_factory
        self._flashcard_workflow: Optional[FlashcardWorkflow]
        if session_factory is None:
            self._flashcard_workflow = None
        else:
            self._flashcard_workflow = FlashcardWorkflow(
                client=self._client,
                model=flashcard_model or self._model,
                session_factory=session_factory,
                source_language=flashcard_source_language,
                target_language=flashcard_target_language,
                max_cards_per_message=flashcard_max_cards,
            )

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
        except Exception:  # pragma: no cover - guardrail against database issues
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

    async def _typing_indicator(
        self, chat_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Continuously send a typing action so users see the bot working."""
        try:
            while True:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            return

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
            await self._store_user_profile(
                chat.id,
                getattr(user, "first_name", None),
                getattr(user, "last_name", None),
            )

        await self._maybe_handle_flashcard_request(update, chat.id, user_message)

        try:
            typing_task = asyncio.create_task(self._typing_indicator(chat.id, context))
            reply = await self._generate_response(chat.id, user_message)
        except Exception:  # pragma: no cover - network or API issues handled gracefully
            LOGGER.exception("Failed to generate response from OpenAI.")
            reply = (
                "Failed to reach the Greek tutor right now. Please try your request again in a moment."
            )
        finally:
            if "typing_task" in locals():
                typing_task.cancel()
                with suppress(asyncio.CancelledError):
                    await typing_task

        if reply:
            await update.message.reply_text(
                reply,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=self._build_take_card_markup(),
            )
            self._record_interaction(chat.id, user_message, reply)

    async def _maybe_handle_flashcard_request(
        self,
        update: Update,
        chat_id: int,
        user_message: str,
    ) -> Optional[FlashcardWorkflowResult]:
        if not update.message:
            return None

        if self._flashcard_workflow is None:
            return None

        result = await self._flashcard_workflow.handle(chat_id, user_message)
        if not result.summaries and not result.errors:
            return result

        acknowledgement = self._format_flashcard_acknowledgement(result)
        if acknowledgement:
            await update.message.reply_text(
                acknowledgement,
                reply_markup=self._build_take_card_markup(),
            )

        return result

    @staticmethod
    def _format_flashcard_acknowledgement(result: FlashcardWorkflowResult) -> str:
        lines: List[str] = []

        if result.summaries:
            lines.append("Карточки готовы:")
            status_suffix = {
                "created": "",
                "reactivated": " (вернул в расписание)",
                "existing": " (уже была)",
            }
            for summary in result.summaries:
                suffix = status_suffix.get(summary.status, "")
                lines.append(f"- {summary.source_text} — {summary.target_text}{suffix}")
                if summary.example:
                    lines.append(f"  Пример: {summary.example}")
            lines.append("Нажми «Взять карточку», когда захочешь потренироваться.")

        if result.errors:
            if lines:
                lines.append("")
            lines.append("Не удалось обработать часть запроса:")
            for error in result.errors:
                lines.append(f"- {error}")

        return "\n".join(lines).strip()

    @staticmethod
    def _build_take_card_markup() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Взять карточку", callback_data="fc_take")]]
        )

    def _build_rating_keyboard(self, user_flashcard_id: int) -> InlineKeyboardMarkup:
        buttons = [
            InlineKeyboardButton(str(score), callback_data=f"fc_rate:{user_flashcard_id}:{score}")
            for score in range(1, 6)
        ]
        return InlineKeyboardMarkup([buttons])

    async def handle_take_flashcard(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        if query is None:
            return

        if self._session_factory is None:
            await query.answer("Карточки временно недоступны.", show_alert=True)
            return

        await query.answer()

        message = query.message
        if message is None or message.chat is None:
            return

        chat_id = message.chat.id

        async with self._session_factory() as session:
            flashcard_record = await get_next_flashcard_for_user(session, chat_id)

        if flashcard_record is None:
            await message.reply_text(
                "Для тебя пока нет карточек. Добавь новые слова, чтобы начать тренировку.",
                reply_markup=self._build_take_card_markup(),
            )
            return

        orientation = random.choice(("source_to_target", "target_to_source"))
        prompt = self._format_flashcard_prompt(flashcard_record, orientation)
        await message.reply_text(
            prompt,
            reply_markup=self._build_rating_keyboard(flashcard_record.id),
        )

    async def handle_rate_flashcard(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return

        if self._session_factory is None:
            await query.answer("Карточки временно недоступны.", show_alert=True)
            return

        parts = query.data.split(":")
        if len(parts) != 3 or parts[0] != "fc_rate":
            await query.answer()
            return

        try:
            user_flashcard_id = int(parts[1])
            score = int(parts[2])
        except ValueError:
            await query.answer("Некорректная оценка.", show_alert=True)
            return

        if score < 1 or score > 5:
            await query.answer("Оценка должна быть от 1 до 5.", show_alert=True)
            return

        message = query.message
        if message is None or message.chat is None:
            await query.answer()
            return

        chat_id = message.chat.id
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            async with session.begin():
                user_flashcard = await session.get(
                    UserFlashcard,
                    user_flashcard_id,
                    options=[selectinload(UserFlashcard.flashcard)],
                )

                if user_flashcard is None or user_flashcard.chat_id != chat_id:
                    await query.answer("Карточка не найдена.", show_alert=True)
                    return

                schedule = calculate_next_schedule(
                    score=score,
                    current_easiness=user_flashcard.easiness_factor,
                    current_interval=user_flashcard.interval,
                    current_repetition=user_flashcard.repetition,
                    now=now,
                )

                await record_flashcard_review(
                    session=session,
                    user_flashcard=user_flashcard,
                    score=score,
                    next_review_at=schedule.next_review_at,
                    easiness_factor=schedule.easiness_factor,
                    interval=schedule.interval,
                    repetition=schedule.repetition,
                    now=now,
                )

        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:  # pragma: no cover - best effort cleanup
            LOGGER.debug("Could not clear flashcard rating markup.", exc_info=True)

        await query.answer("Оценка сохранена.")

        await message.reply_text(
            f"Оценка {score} сохранена. Вернёмся к карточке {self._describe_interval(schedule.interval)}.",
            reply_markup=self._build_take_card_markup(),
        )

    @staticmethod
    def _format_flashcard_prompt(
        record: UserFlashcard,
        orientation: str,
    ) -> str:
        flashcard = record.flashcard
        if flashcard is None:
            return "Не удалось загрузить карточку."

        if orientation == "target_to_source":
            first_label, first_value = "Перевод", flashcard.target_text
            second_label, second_value = "Слово", flashcard.source_text
            header = "Вспомни исходное слово:"
        else:
            first_label, first_value = "Слово", flashcard.source_text
            second_label, second_value = "Перевод", flashcard.target_text
            header = "Вспомни перевод:"

        lines = [
            "🃏 Карточка",
            header,
            f"{first_label}: {first_value}",
            f"{second_label}: {second_value}",
        ]

        if flashcard.example:
            lines.append(f"Пример: {flashcard.example}")

        lines.append("")
        lines.append("Как хорошо ты помнишь это слово? Выбери оценку:")
        return "\n".join(lines).strip()

    @staticmethod
    def _describe_interval(interval_days: int) -> str:
        if interval_days <= 0:
            return "очень скоро"
        if interval_days == 1:
            return "через 1 день"
        if interval_days < 7:
            return f"через {interval_days} дня"
        if interval_days % 7 == 0:
            weeks = interval_days // 7
            if weeks == 1:
                return "через 1 неделю"
            return f"через {weeks} недель"
        return f"через {interval_days} дней"

