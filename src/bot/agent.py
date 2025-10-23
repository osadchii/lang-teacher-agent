"""Telegram handlers for the Greek language teacher agent."""

from __future__ import annotations

import asyncio
import base64
import csv
import imghdr
import json
import logging
import random
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
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
from src.db.flashcards import (
    FlashcardPayload,
    ensure_user_flashcard,
    get_next_flashcard_for_user,
    get_or_create_flashcard,
    record_flashcard_review,
)
from src.db.users import (
    UserStatistics,
    get_user_statistics,
    increment_user_statistics,
    upsert_user,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ImageTextResult:
    """Structured result for OCR + translation of Greek text found in images."""

    greek_text: str
    translation: str
    words: List[Tuple[str, str]]

    @property
    def word_count(self) -> int:
        return len(self.words)


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
        flashcard_max_cards: int = 10,
        vision_model: Optional[str] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._vision_model = vision_model or model
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
        self._common_words_path = Path(__file__).resolve().parents[2] / "static" / "greek_top_500_words.csv"
        self._common_words_cache: Optional[List[FlashcardPayload]] = None

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

    def _load_common_flashcards(self) -> List[FlashcardPayload]:
        """Load and cache flashcard payloads for the 500 most common words."""
        if self._common_words_cache is not None:
            return self._common_words_cache

        entries: List[FlashcardPayload] = []
        try:
            with self._common_words_path.open("r", encoding="utf-8-sig") as csv_file:
                reader = csv.reader(csv_file, delimiter=";")
                next(reader, None)  # discard header
                for row_index, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    source = row[0].strip()
                    target = row[1].strip() if len(row) > 1 else ""
                    example = row[2].strip() if len(row) > 2 else ""
                    if not source or not target:
                        LOGGER.debug(
                            "Skipping incomplete common words row %s: %s",
                            row_index,
                            row,
                        )
                        continue
                    entries.append(
                        FlashcardPayload(
                            source_text=source,
                            target_text=target,
                            example=example or None,
                            source_lang=self._flashcard_source_language,
                            target_lang=self._flashcard_target_language,
                        )
                    )
        except FileNotFoundError as exc:
            raise RuntimeError("Common words CSV file is missing.") from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise RuntimeError("Failed to load common words CSV file.") from exc

        self._common_words_cache = entries
        return entries

    async def _download_photo_bytes(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        file_id: str,
    ) -> bytes:
        """Download a Telegram photo by file_id and return its raw bytes."""
        telegram_file = await context.bot.get_file(file_id)
        buffer = BytesIO()
        await telegram_file.download_to_memory(out=buffer)
        return buffer.getvalue()

    async def _extract_greek_text_from_image(
        self,
        image_bytes: bytes,
        chat_id: Optional[int] = None,
    ) -> Optional[ImageTextResult]:
        """Send an image to OpenAI and parse the detected Greek text plus translation."""
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        image_format = imghdr.what(None, image_bytes)
        if image_format == "jpeg":
            mime_type = "image/jpeg"
        elif image_format == "png":
            mime_type = "image/png"
        elif image_format == "gif":
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"
        data_url = f"data:{mime_type};base64,{encoded_image}"

        try:
            response = await self._client.responses.create(
                model=self._vision_model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You extract Modern Greek text from images and translate it into Russian. "
                                    "Always respond with strict JSON using the schema: "
                                    '{"status": "ok" | "no_greek", "greek_text": string, "translation": string, '
                                    '"words": [{"greek": string, "translation": string}]}. '
                                    "Treat only characters from the Greek alphabet (including tonos) as Greek text. "
                                    "If there is no Greek text, use status \"no_greek\" and leave other fields empty. "
                                    "Limit the words list to at most 10 unique Greek words in the order they appear, "
                                    "and provide their Russian translations when possible."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Найди весь греческий текст на изображении, выпиши его без изменений и переведи на русский.",
                            },
                            {"type": "input_image", "image_url": data_url},
                        ],
                    },
                ],
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.exception("Failed OCR request%s.", f" for chat {chat_id}" if chat_id else "")
            raise RuntimeError("Failed to request image recognition.") from exc

        raw_text = extract_output_text(response).strip()
        LOGGER.debug("OCR raw response%s: %s", f" for chat {chat_id}" if chat_id else "", raw_text[:500])
        cleaned = self._strip_code_fences(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Malformed OCR response: %s", raw_text)
            raise RuntimeError("Image recognition returned malformed data.") from exc

        status = str(payload.get("status") or "").lower()
        greek_text = (payload.get("greek_text") or "").strip()
        translation = (payload.get("translation") or "").strip()

        if not greek_text:
            candidate = payload.get("text") or payload.get("raw_text")
            if isinstance(candidate, str):
                greek_text = candidate.strip()

        if not greek_text:
            lines_field = payload.get("lines")
            if isinstance(lines_field, list):
                collected: List[str] = []
                for entry in lines_field:
                    if isinstance(entry, str):
                        stripped_entry = entry.strip()
                        if stripped_entry and self._contains_greek_characters(stripped_entry):
                            collected.append(stripped_entry)
                if collected:
                    greek_text = "\n".join(collected)

        if status == "no_greek":
            LOGGER.info(
                "OCR reported no Greek text%s. keys=%s",
                f" for chat {chat_id}" if chat_id else "",
                list(payload.keys()),
            )
            return None

        if not greek_text or not self._contains_greek_characters(greek_text):
            LOGGER.info(
                "OCR could not confirm Greek text%s. status=%s detected_text_preview=%r",
                f" for chat {chat_id}" if chat_id else "",
                status or "missing",
                (greek_text or "")[:120],
            )
            return None

        raw_words = payload.get("words") or []
        words: List[Tuple[str, str]] = []
        for entry in raw_words:
            if not isinstance(entry, dict):
                continue
            greek_word = str(entry.get("greek") or "").strip()
            translation_word = str(entry.get("translation") or "").strip()
            if greek_word:
                words.append((greek_word, translation_word))
            if len(words) >= 10:
                break

        LOGGER.info(
            "OCR success%s. status=%s greek_chars=%d translation_chars=%d words=%d",
            f" for chat {chat_id}" if chat_id else "",
            status or "missing",
            len(greek_text),
            len(translation),
            len(words),
        )

        return ImageTextResult(
            greek_text=greek_text,
            translation=translation,
            words=words,
        )

    @staticmethod
    def _strip_code_fences(response_text: str) -> str:
        """Remove optional Markdown fences around JSON responses."""
        stripped = response_text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            body = stripped.split("\n", 1)
            if len(body) == 2:
                content = body[1].rsplit("\n", 1)
                if len(content) == 2:
                    return content[0]
        return stripped

    @staticmethod
    def _contains_greek_characters(text: str) -> bool:
        """Check if the provided text includes any Greek alphabet characters."""
        for char in text:
            code_point = ord(char)
            if 0x0370 <= code_point <= 0x03FF or 0x1F00 <= code_point <= 0x1FFF:
                return True
        return False

    def _format_image_result_message(self, result: ImageTextResult) -> str:
        """Compose an HTML response summarizing OCR findings and translation."""
        lines: List[str] = [
            "<b>Найденный греческий текст:</b>",
            f"<pre>{self._escape_html(result.greek_text)}</pre>",
            "<b>Перевод:</b>",
            self._escape_html(result.translation or "Перевод нужно уточнить."),
        ]

        if result.words:
            lines.append("")
            lines.append("<b>Слова из текста:</b>")
            for greek_word, russian_word in result.words:
                if russian_word:
                    lines.append(f"- {self._escape_html(greek_word)} - {self._escape_html(russian_word)}")
                else:
                    lines.append(f"- {self._escape_html(greek_word)} - перевод нужно уточнить")

        if 0 < result.word_count <= 5:
            lines.append("")
            lines.append(
                "Если хочешь, помогу добавить эти слова в карточки (до 10 за раз) — просто напиши."
            )

        return "\n".join(lines).strip()

    async def _ensure_common_flashcards(
        self,
        session: AsyncSession,
        chat_id: int,
        *,
        now: Optional[datetime] = None,
    ) -> Tuple[int, int, int, int]:
        """Ensure the shared and user-specific flashcards exist for the common words."""
        if now is None:
            now = datetime.now(timezone.utc)

        payloads = self._load_common_flashcards()
        created_flashcards = 0
        new_user_links = 0
        already_had_links = 0
        reactivated_links = 0

        for payload in payloads:
            flashcard, created = await get_or_create_flashcard(session, payload)
            if created:
                created_flashcards += 1

            _, user_created, reactivated = await ensure_user_flashcard(
                session=session,
                chat_id=chat_id,
                flashcard=flashcard,
                now=now,
            )
            if user_created:
                new_user_links += 1
            else:
                already_had_links += 1
            if reactivated:
                reactivated_links += 1

        return created_flashcards, new_user_links, already_had_links, reactivated_links

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

    async def _increment_user_stats(
        self,
        chat_id: int,
        *,
        questions: int = 0,
        added: int = 0,
        reviewed: int = 0,
        mastered: int = 0,
    ) -> None:
        """Increment user statistics counters, ignoring errors."""
        if self._session_factory is None:
            return

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await increment_user_statistics(
                        session,
                        chat_id,
                        questions=questions,
                        added=added,
                        reviewed=reviewed,
                        mastered=mastered,
                    )
        except Exception:  # pragma: no cover - guardrail against database issues
            LOGGER.exception("Failed to update statistics for chat %s.", chat_id)

    async def _fetch_user_statistics(self, chat_id: int) -> Optional[UserStatistics]:
        """Load persisted statistics for a user, returning None on failure."""
        if self._session_factory is None:
            return None

        try:
            async with self._session_factory() as session:
                return await get_user_statistics(session, chat_id)
        except Exception:  # pragma: no cover - guardrail against database issues
            LOGGER.exception("Failed to load statistics for chat %s.", chat_id)
            return None

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
            await self._increment_user_stats(chat.id, questions=1)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process a photo, extract Greek text, and return a translation."""
        message = update.message
        if not message or not message.photo:
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

        largest_photo = message.photo[-1]
        try:
            image_bytes = await self._download_photo_bytes(context, largest_photo.file_id)
        except Exception:  # pragma: no cover - network errors
            LOGGER.exception("Failed to download photo for chat %s.", chat.id)
            await message.reply_text("Не удалось получить изображение. Попробуй отправить фото ещё раз.")
            return

        typing_task = asyncio.create_task(self._typing_indicator(chat.id, context))
        recognition_failed = False
        result: Optional[ImageTextResult] = None

        try:
            result = await self._extract_greek_text_from_image(image_bytes, chat_id=chat.id)
        except RuntimeError:
            recognition_failed = True
            LOGGER.exception("Image recognition failed for chat %s.", chat.id)
        finally:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task

        if recognition_failed:
            reply_text = "Не получилось распознать текст на этом фото. Попробуй ещё раз немного позже."
            await message.reply_text(reply_text)
            self._record_interaction(chat.id, "[Фото]", reply_text)
            await self._increment_user_stats(chat.id, questions=1)
            return

        if result is None:
            reply_text = "Похоже, на фото нет греческого текста. Пришли изображение, где текст именно на греческом."
            await message.reply_text(reply_text)
            self._record_interaction(chat.id, "[Фото]", reply_text)
            await self._increment_user_stats(chat.id, questions=1)
            return

        reply_text = self._format_image_result_message(result)
        await message.reply_text(
            reply_text,
            parse_mode=ParseMode.HTML,
            reply_markup=self._build_take_card_markup(),
        )
        self._record_interaction(chat.id, "[Фото]", reply_text)
        await self._increment_user_stats(chat.id, questions=1)

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

    async def handle_stat(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Present a formatted progress dashboard for the current user."""
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

        stats = await self._fetch_user_statistics(chat.id)
        if stats is None:
            await update.message.reply_text("Статистика появится после первых занятий.")
            return

        created_at = stats.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        days_active = (now.date() - created_at.date()).days + 1
        if days_active < 1:
            days_active = 1

        avg_per_day = stats.flashcards_reviewed / days_active if stats.flashcards_reviewed else 0.0
        display_name = stats.first_name or stats.last_name or "студент"

        message_lines = [
            "📊 <b>Статистика обучения</b>",
            f"👤 {display_name}",
            f"🗓 С {created_at.strftime('%d.%m.%Y')}",
            "",
            f"❓ <b>Вопросов преподавателю:</b> {stats.questions_asked}",
            f"🆕 <b>Добавлено карточек:</b> {stats.flashcards_added}",
            f"🔁 <b>Повторено карточек:</b> {stats.flashcards_reviewed}",
            f"🌟 <b>Отлично выучено (5/5):</b> {stats.flashcards_mastered}",
            f"⚡️ <b>Средний темп:</b> {avg_per_day:.2f} в день",
        ]

        await update.message.reply_text(
            "\n".join(message_lines),
            parse_mode=ParseMode.HTML,
        )

    async def handle_add_500_common_words(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Import shared flashcards for the most common Greek words and attach them to the user."""
        if not update.message:
            return

        chat = update.effective_chat
        if chat is None:
            return

        if self._session_factory is None:
            await update.message.reply_text("Хранилище карточек недоступно в этой конфигурации.")
            return

        user = update.effective_user
        if user is not None:
            await self._store_user_profile(
                chat.id,
                getattr(user, "first_name", None),
                getattr(user, "last_name", None),
            )

        try:
            payloads = self._load_common_flashcards()
        except Exception:
            LOGGER.exception("Failed to load the top 500 Greek words from CSV.")
            await update.message.reply_text(
                "Не удалось загрузить список популярных слов. Попробуй позже."
            )
            return

        total_words = len(payloads)
        now = datetime.now(timezone.utc)

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    (
                        created_count,
                        added_count,
                        already_had_count,
                        reactivated_count,
                    ) = await self._ensure_common_flashcards(
                        session,
                        chat.id,
                        now=now,
                    )
        except Exception:
            LOGGER.exception("Failed to persist common words for chat %s.", chat.id)
            await update.message.reply_text(
                "Не удалось добавить карточки. Попробуй позже."
            )
            return

        LOGGER.info(
            "Common word import for chat %s completed: created=%s, new_links=%s, already_had=%s, reactivated=%s.",
            chat.id,
            created_count,
            added_count,
            already_had_count,
            reactivated_count,
        )

        if added_count:
            await self._increment_user_stats(chat.id, added=added_count)

        if added_count:
            intro = f"Список из {total_words} слов готов!"
        else:
            intro = f"Все {total_words} слов уже были в твоей коллекции!"

        message_lines = [intro]
        if added_count:
            message_lines.append(f"Добавлено новых карточек: {added_count}.")
        if added_count and already_had_count:
            message_lines.append(f"Ещё {already_had_count} уже были у тебя.")
        message_lines.append("Можешь тренироваться, нажав «Взять карточку».")

        await update.message.reply_text(
            "\n".join(message_lines),
            reply_markup=self._build_take_card_markup(),
        )

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
        created_cards = sum(1 for summary in result.summaries if summary.status == "created")
        if created_cards:
            await self._increment_user_stats(chat_id, added=created_cards)
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
                f"- {self._escape_html(summary.source_text)} — {self._escape_html(summary.target_text)}"
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

        from_user = getattr(query, "from_user", None)
        if from_user is not None:
            await self._store_user_profile(
                chat_id,
                getattr(from_user, "first_name", None),
                getattr(from_user, "last_name", None),
            )

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
                await increment_user_statistics(
                    session,
                    chat_id,
                    reviewed=1,
                    mastered=1 if score == 5 else 0,
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

