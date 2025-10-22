"""Telegram handlers for the Greek language teacher agent."""

from __future__ import annotations

import asyncio
import logging
import random
from html import escape
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

from src.bot.flashcard_workflow import (
    FlashcardSummary,
    FlashcardWorkflow,
    FlashcardWorkflowResult,
)
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
        self._flashcard_source_language = flashcard_source_language
        self._flashcard_target_language = flashcard_target_language
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

        flashcard_result = await self._maybe_handle_flashcard_request(update, chat.id, user_message)
        if flashcard_result and flashcard_result.handled:
            return

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

    async def handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Send a welcome message in Russian and Greek describing the bot's abilities."""
        if not update.message:
            return

        chat = update.effective_chat
        if chat is None:
            return

        user = update.effective_user
        if user is not None:
            await self._store_user_profile(
                chat.id,
                getattr(user, "first_name", None),
                getattr(user, "last_name", None),
            )

        greeting = (
            "Привет! Я твой виртуальный преподаватель греческого.\n"
            "Вот чем могу помочь:\n"
            "- объяснять правила греческого языка;\n"
            "- помогать с переводами слов и фраз;\n"
            "- тренироваться с флеш-карточками для запоминания слов.\n\n"
            "Γεια σου! Είμαι ο ψηφιακός σου καθηγητής ελληνικών.\n"
            "Μπορώ να βοηθήσω με:\n"
            "- επεξήγηση γραμματικών κανόνων στα ελληνικά,\n"
            "- μεταφράσεις λέξεων και φράσεων,\n"
            "- εξάσκηση με κάρτες μνήμης για να θυμάσαι λεξιλόγιο."
        )

        await update.message.reply_text(greeting, parse_mode=ParseMode.MARKDOWN)

    def _build_flashcard_context(self, chat_id: int) -> Optional[str]:
        """Return the most recent exchange to help flashcard extraction."""
        history = self._get_history(chat_id)
        if not history:
            return None

        previous_user, previous_reply = history[-1]
        lines: List[str] = []
        if previous_user:
            lines.append(f"Предыдущее сообщение пользователя: {previous_user.strip()}")
        if previous_reply:
            lines.append(f"Ответ преподавателя: {previous_reply.strip()}")
        if not lines:
            return None
        lines.append("Добавь подходящую лексику из этого ответа в карточки ученика.")
        return "\n".join(lines)

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

        context_payload = self._build_flashcard_context(chat_id)
        progress_message = None
        if self._flashcard_workflow.is_probable_request(user_message):
            with suppress(Exception):
                progress_message = await update.message.reply_text("Готовлю карточки...")

        result = await self._flashcard_workflow.handle(
            chat_id,
            user_message,
            context=context_payload,
        )
        if not result.summaries and not result.errors:
            if progress_message is not None:
                fallback_text = result.reason or "Карточки не распознаны."
                try:
                    await progress_message.edit_text(fallback_text)
                except Exception:  # pragma: no cover - best effort status update
                    LOGGER.debug("Could not update flashcard progress message.", exc_info=True)
            return result

        new_cards = [
            summary for summary in result.summaries if summary.status in {"created", "reactivated"}
        ]
        existing_cards = [
            summary for summary in result.summaries if summary.status not in {"created", "reactivated"}
        ]

        progress_consumed = False

        async def deliver_response(
            text: Optional[str],
            *,
            parse_mode: Optional[str] = None,
            markup: Optional[InlineKeyboardMarkup] = None,
        ) -> bool:
            nonlocal progress_message, progress_consumed
            if not text:
                return False

            if progress_message is not None and not progress_consumed:
                try:
                    await progress_message.edit_text(
                        text,
                        parse_mode=parse_mode,
                        reply_markup=markup,
                    )
                    progress_consumed = True
                    return True
                except Exception:  # pragma: no cover - best effort status update
                    LOGGER.debug("Could not edit flashcard progress message.", exc_info=True)
                    with suppress(Exception):
                        await progress_message.delete()
                    progress_message = None

            await update.message.reply_text(
                text,
                parse_mode=parse_mode,
                reply_markup=markup,
            )
            return True

        if new_cards:
            for summary in new_cards:
                await deliver_response(
                    self._format_flashcard_added_message(summary),
                    parse_mode=ParseMode.HTML,
                    markup=self._build_flashcard_added_markup(summary.user_flashcard_id),
                )

            await update.message.reply_text(
                "Когда будешь готов потренироваться, нажми «Взять карточку».",
                reply_markup=self._build_take_card_markup(),
            )

        if existing_cards:
            await deliver_response(
                self._format_existing_flashcards(existing_cards),
                parse_mode=ParseMode.HTML,
            )

        if result.errors:
            await deliver_response(
                self._format_flashcard_errors(result.errors, result.reason),
                parse_mode=ParseMode.HTML,
            )

        if not progress_consumed and progress_message is not None:
            with suppress(Exception):
                await progress_message.delete()

        return result

    def _format_flashcard_added_message(self, summary: FlashcardSummary) -> str:
        if summary.status == "reactivated":
            title = "Карточка снова активна"
        else:
            title = "Карточка добавлена"

        lines = [
            f"<b>{self._escape_html(title)}</b>",
            f"<b>{self._escape_html(self._flashcard_source_language)}:</b> {self._escape_html(summary.source_text)}",
            f"<b>{self._escape_html(self._flashcard_target_language)}:</b> {self._escape_html(summary.target_text)}",
        ]

        if summary.example:
            lines.append(f"<i>Пример:</i> {self._escape_html(summary.example)}")

        lines.append("")
        lines.append("Если хочешь отменить добавление, нажми кнопку ниже.")
        return "\n".join(lines).strip()

    def _build_flashcard_added_markup(
        self,
        user_flashcard_id: Optional[int],
    ) -> Optional[InlineKeyboardMarkup]:
        if user_flashcard_id is None:
            return None

        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Удалить карточку",
                        callback_data=f"fc_delete:{user_flashcard_id}",
                    )
                ]
            ]
        )

    def _format_existing_flashcards(self, summaries: List[FlashcardSummary]) -> str:
        if not summaries:
            return ""

        lines = ["<b>Эти карточки уже есть у тебя:</b>"]
        for summary in summaries:
            lines.append(
                f"- {self._escape_html(summary.source_text)} &mdash; {self._escape_html(summary.target_text)}"
            )
        return "\n".join(lines).strip()

    def _format_flashcard_errors(
        self,
        errors: List[str],
        reason: Optional[str],
    ) -> str:
        lines: List[str] = []

        if reason:
            lines.append(self._escape_html(reason))

        if errors:
            lines.append("<b>Не получилось обработать карточки:</b>")
            for error in errors:
                lines.append(f"- {self._escape_html(error)}")

        return "\n".join(lines).strip()

    def _format_flashcard_deleted_message(
        self,
        source_text: str,
        target_text: str,
    ) -> str:
        lines = [
            "<b>Карточка удалена</b>",
            f"<b>{self._escape_html(self._flashcard_source_language)}:</b> {self._escape_html(source_text)}",
            f"<b>{self._escape_html(self._flashcard_target_language)}:</b> {self._escape_html(target_text)}",
        ]

        lines.append("")
        lines.append("Эта карточка больше не будет предлагаться.")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_take_card_markup() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Взять карточку", callback_data="fc_take")]]
        )

    @staticmethod
    def _build_reveal_keyboard(user_flashcard_id: int, orientation: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Показать ответ",
                        callback_data=f"fc_show:{user_flashcard_id}:{orientation}",
                    )
                ]
            ]
        )

    def _build_rating_keyboard(self, user_flashcard_id: int) -> InlineKeyboardMarkup:
        buttons = [
            InlineKeyboardButton(str(score), callback_data=f"fc_rate:{user_flashcard_id}:{score}")
            for score in range(1, 6)
        ]
        return InlineKeyboardMarkup([buttons])

    async def handle_delete_flashcard(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return

        if self._session_factory is None:
            await query.answer("Функция недоступна.", show_alert=True)
            return

        parts = query.data.split(":")
        if len(parts) != 2 or parts[0] != "fc_delete":
            await query.answer()
            return

        try:
            user_flashcard_id = int(parts[1])
        except ValueError:
            await query.answer("Некорректный идентификатор.", show_alert=True)
            return

        message = query.message
        if message is None or message.chat is None:
            await query.answer()
            return

        chat_id = message.chat.id

        source_text = ""
        target_text = ""

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

                flashcard = user_flashcard.flashcard
                if flashcard is None:
                    await query.answer("Карточка не найдена.", show_alert=True)
                    return

                source_text = flashcard.source_text or ""
                target_text = flashcard.target_text or ""

                await session.delete(user_flashcard)

        await query.answer("Карточка удалена.")

        confirmation = self._format_flashcard_deleted_message(source_text, target_text)

        try:
            await query.edit_message_text(
                confirmation,
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:  # pragma: no cover - best effort update
            LOGGER.debug("Could not update flashcard deletion message.", exc_info=True)
            with suppress(Exception):
                await message.reply_text(
                    confirmation,
                    parse_mode=ParseMode.HTML,
                )

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
        prompt = self._format_flashcard_question(flashcard_record, orientation)
        await message.reply_text(
            prompt,
            parse_mode=ParseMode.HTML,
            reply_markup=self._build_reveal_keyboard(flashcard_record.id, orientation),
        )

    async def handle_show_flashcard(
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
        if len(parts) != 3 or parts[0] != "fc_show":
            await query.answer()
            return

        try:
            user_flashcard_id = int(parts[1])
        except ValueError:
            await query.answer("Некорректный запрос.", show_alert=True)
            return

        orientation = parts[2]
        if orientation not in {"source_to_target", "target_to_source"}:
            await query.answer("Не удалось определить направление карточки.", show_alert=True)
            return

        message = query.message
        if message is None or message.chat is None:
            await query.answer()
            return

        chat_id = message.chat.id

        async with self._session_factory() as session:
            user_flashcard = await session.get(
                UserFlashcard,
                user_flashcard_id,
                options=[selectinload(UserFlashcard.flashcard)],
            )

        if user_flashcard is None or user_flashcard.chat_id != chat_id:
            await query.answer("Карточка не найдена.", show_alert=True)
            return

        prompt = self._format_flashcard_prompt(user_flashcard, orientation)

        try:
            await query.edit_message_text(
                prompt,
                parse_mode=ParseMode.HTML,
                reply_markup=self._build_rating_keyboard(user_flashcard.id),
            )
        except Exception:  # pragma: no cover - best effort update
            LOGGER.debug("Could not reveal flashcard text.", exc_info=True)
            with suppress(Exception):
                await query.edit_message_reply_markup(reply_markup=None)
            await message.reply_text(
                prompt,
                parse_mode=ParseMode.HTML,
                reply_markup=self._build_rating_keyboard(user_flashcard.id),
            )

        await query.answer()

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
    def _escape_html(text: Optional[str]) -> str:
        if not text:
            return ''
        return escape(text, quote=False)

    def _format_flashcard_question(
        self,
        record: UserFlashcard,
        orientation: str,
    ) -> str:
        flashcard = record.flashcard
        if flashcard is None:
            return 'Карточку не удалось загрузить.'

        source_label = 'Исходная фраза'
        target_label = 'Перевод'
        direction = f"{self._flashcard_source_language} → {self._flashcard_target_language}"
        task_language = self._flashcard_target_language
        visible_label = source_label
        visible_value = self._escape_html(flashcard.source_text)

        if orientation == 'target_to_source':
            direction = f"{self._flashcard_target_language} → {self._flashcard_source_language}"
            task_language = self._flashcard_source_language
            visible_label = target_label
            visible_value = self._escape_html(flashcard.target_text)

        lines = [
            '<b>Новая карточка</b>',
            f"<i>Направление:</i> {self._escape_html(direction)}",
            '',
            f"<b>Задание:</b> Переведи на {self._escape_html(task_language)}.",
            f"<b>{self._escape_html(visible_label)}:</b> {visible_value}",
            '',
            '<i>Нажми «Показать ответ», когда будешь готов.</i>',
        ]
        return "\n".join(lines).strip()

    def _format_flashcard_prompt(
        self,
        record: UserFlashcard,
        orientation: str,
    ) -> str:
        flashcard = record.flashcard
        if flashcard is None:
            return 'Карточку не удалось показать.'

        source_label = 'Исходная фраза'
        target_label = 'Перевод'
        prompt_label = target_label
        prompt_value = self._escape_html(flashcard.target_text)
        answer_label = source_label
        answer_value = self._escape_html(flashcard.source_text)

        if orientation == 'source_to_target':
            prompt_label = source_label
            prompt_value = self._escape_html(flashcard.source_text)
            answer_label = target_label
            answer_value = self._escape_html(flashcard.target_text)

        lines = [
            '<b>Ответ по карточке</b>',
            f"<b>{self._escape_html(prompt_label)}:</b> {prompt_value}",
            f"<b>{self._escape_html(answer_label)}:</b> {answer_value}",
        ]

        if flashcard.example:
            lines.append(f"<i>Пример:</i> {self._escape_html(flashcard.example)}")

        lines.append('')
        lines.append('<i>Оцени карточку, чтобы продолжить.</i>')
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

