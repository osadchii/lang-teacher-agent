"""Telegram handlers for the Greek language teacher agent."""

from __future__ import annotations

import asyncio
import csv
import logging
import random
import re
from contextlib import suppress
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from src.bot.agent_components.image_support import (
    ImageTextResult,
    analyze_image_words,
    build_photo_prompt,
    collect_unique_image_words,
    download_photo_bytes,
    extract_greek_text_from_image,
    format_image_flashcard_section,
    format_image_result_message,
    select_image_words_for_review,
)
from src.bot.flashcard_workflow import (
    FlashcardSummary,
    FlashcardWorkflow,
    FlashcardWorkflowResult,
)
from src.bot.message_history import MessageHistoryService
from src.bot.openai_utils import extract_output_text
from src.bot.srs import calculate_next_schedule
from src.db import UserFlashcard
from src.db.flashcards import (
    FlashcardPayload,
    ensure_user_flashcard,
    get_next_flashcard_for_user,
    get_or_create_flashcard,
    get_user_flashcard_by_source_text,
    record_flashcard_review,
)
from src.db.users import (
    UserStatistics,
    get_user_statistics,
    increment_user_statistics,
    upsert_user,
)


LOGGER = logging.getLogger(__name__)

_GREEK_SEQUENCE_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]+(?:[\s\-][\u0370-\u03ff\u1f00-\u1fff]+)*")
_GREEK_ARTICLE_PREFIXES = (
    "ο",
    "η",
    "το",
    "οι",
    "τα",
    "τον",
    "την",
    "τους",
    "τις",
    "των",
    "του",
    "της",
)


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
        self._history = MessageHistoryService(history_size)
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
        self._common_words_lookup: Optional[Set[str]] = None
        self._article_lookup: Set[str] = {self._normalize_greek_term(article) for article in _GREEK_ARTICLE_PREFIXES}
        self._pending_flashcard_terms: Dict[int, Union[str, List[str]]] = {}

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
        self._common_words_lookup = {
            self._normalize_greek_term(payload.source_text)
            for payload in entries
            if payload.source_text
        }
        return entries
    
    @staticmethod
    def _normalize_greek_term(term: str) -> str:
        cleaned = "".join(ch for ch in term.casefold() if ch.isalnum() or ch.isspace())
        return " ".join(cleaned.split())

    def _build_article_variants(self, term: str) -> List[str]:
        """Return normalized variants of term with and without Greek articles."""
        variants: List[str] = []
        raw = (term or "").strip()
        if raw:
            variants.append(raw)

        normalized = self._normalize_greek_term(raw)
        if normalized:
            variants.append(normalized)
            tokens = normalized.split()
            if tokens and tokens[0] in _GREEK_ARTICLE_PREFIXES and len(tokens) > 1:
                without_article = " ".join(tokens[1:])
                if without_article:
                    variants.append(without_article)
            else:
                for article in _GREEK_ARTICLE_PREFIXES:
                    variants.append(f"{article} {normalized}")

        deduped: List[str] = []
        seen: set[str] = set()
        for variant in variants:
            collapsed = " ".join(variant.split())
            if not collapsed or collapsed in seen:
                continue
            seen.add(collapsed)
            deduped.append(collapsed)
        return deduped

    @staticmethod
    def _extract_greek_segments(text: Optional[str]) -> List[str]:
        if not text:
            return []
        return [match.group(0).strip() for match in _GREEK_SEQUENCE_RE.finditer(text)]

    def _is_probable_proper_noun(self, term: str, common_lookup: Set[str]) -> bool:
        stripped = (term or "").strip()
        if not stripped:
            return False

        first_char = stripped[0]
        if not first_char.isalpha() or not first_char.isupper():
            return False

        normalized = self._normalize_greek_term(stripped)
        if not normalized:
            return False

        if normalized in self._article_lookup:
            return False

        if normalized in common_lookup:
            return False

        return True

    def _get_common_word_lookup(self) -> Set[str]:
        if self._common_words_lookup is None:
            try:
                entries = self._load_common_flashcards()
            except Exception:  # pragma: no cover - defensive fallback
                LOGGER.debug("Failed to load common words lookup.", exc_info=True)
                self._common_words_lookup = set()
            else:
                self._common_words_lookup = {
                    self._normalize_greek_term(entry.source_text)
                    for entry in entries
                    if entry.source_text
                }
        return self._common_words_lookup

    def _select_primary_term(self, candidates: List[str]) -> Optional[str]:
        for term in candidates:
            normalized = self._normalize_greek_term(term)
            if not normalized:
                continue
            if len(normalized.split()) <= 3:
                return term.strip()
        for term in candidates:
            normalized = self._normalize_greek_term(term)
            if normalized:
                return term.strip()
        return None

    def _extract_primary_translation_term(self, user_message: str, assistant_reply: str) -> Optional[str]:
        message_terms = self._extract_greek_segments(user_message)
        normalized_lookup: Dict[str, str] = {}
        for term in message_terms:
            normalized = self._normalize_greek_term(term)
            if not normalized or normalized in normalized_lookup:
                continue
            normalized_lookup[normalized] = term.strip()
        if len(normalized_lookup) == 1:
            return next(iter(normalized_lookup.values()))

        if assistant_reply:
            for line in assistant_reply.splitlines():
                lowered = line.casefold()
                if "пример" in lowered:
                    continue
                if "произнош" in lowered:
                    continue
                if "по-гречески" in lowered or "греч." in lowered:
                    terms = self._extract_greek_segments(line)
                    primary = self._select_primary_term(terms)
                    if primary:
                        return primary

            dash_match = re.search(
                r"[—:-]\s*([\u0370-\u03ff\u1f00-\u1fff][\u0370-\u03ff\u1f00-\u1fff\s\-]*)",
                assistant_reply,
            )
            if dash_match:
                snippet = dash_match.group(1)
                terms = self._extract_greek_segments(snippet)
                primary = self._select_primary_term(terms)
                if primary:
                    return primary

            terms = self._extract_greek_segments(assistant_reply)
            primary = self._select_primary_term(terms)
            if primary:
                return primary

        return None

    async def _user_has_active_flashcard(self, chat_id: int, source_text: str) -> bool:
        if self._session_factory is None:
            return False

        candidates = self._build_article_variants(source_text)
        if not candidates:
            return False

        try:
            async with self._session_factory() as session:
                for variant in candidates:
                    record = await get_user_flashcard_by_source_text(
                        session,
                        chat_id,
                        variant,
                        self._flashcard_source_language,
                    )
                    if record is None:
                        continue
                    if record.is_active:
                        return True
        except Exception:
            LOGGER.exception("Failed to check existing flashcards for chat %s.", chat_id)
            return True

        return False

    async def _resolve_reply_markup(
        self,
        chat_id: int,
        user_message: str,
        assistant_reply: str,
    ) -> Tuple[InlineKeyboardMarkup, Optional[str]]:
        if self._flashcard_workflow is None or self._session_factory is None:
            self._pending_flashcard_terms.pop(chat_id, None)
            return self._build_take_card_markup(), None

        primary_term = self._extract_primary_translation_term(user_message, assistant_reply)
        if not primary_term:
            self._pending_flashcard_terms.pop(chat_id, None)
            return self._build_take_card_markup(), None

        if await self._user_has_active_flashcard(chat_id, primary_term):
            self._pending_flashcard_terms.pop(chat_id, None)
            notice = "Это слово уже есть в твоих карточках — потренируйся, чтобы закрепить значение!"
            return self._build_take_card_markup(), notice

        self._pending_flashcard_terms[chat_id] = primary_term
        return self._build_add_card_markup(), None

    @staticmethod
    def _strip_add_card_prompts(reply: str) -> str:
        if not reply:
            return reply

        removal_keywords = ("добав", "созд", "сохран", "занес", "перенес", "flashcard")
        filtered_lines: List[str] = []
        previous_blank = False
        for raw_line in reply.splitlines():
            lowered = raw_line.casefold()
            if ("карточ" in lowered and any(keyword in lowered for keyword in removal_keywords)) or "добав" in lowered:
                continue
            if not raw_line.strip():
                if previous_blank:
                    continue
                previous_blank = True
            else:
                previous_blank = False
            filtered_lines.append(raw_line)

        cleaned = "\n".join(filtered_lines).strip()
        return cleaned

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

    async def _generate_response(self, chat_id: int, user_message: str) -> str:
        """Call the OpenAI Responses API and return plain text."""
        response = await self._client.responses.create(
            model=self._model,
            input=self._history.build_messages(self._system_prompt, chat_id, user_message),
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

        self._pending_flashcard_terms.pop(chat.id, None)

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

        flashcard_result = await self._maybe_handle_flashcard_request(update.message, chat.id, user_message)
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
            reply = self._strip_add_card_prompts(reply)
            markup, notice = await self._resolve_reply_markup(chat.id, user_message, reply)
            if notice:
                if reply:
                    reply = f"{reply}\n\n{notice}"
                else:
                    reply = notice
            elif chat.id in self._pending_flashcard_terms:
                invitation = "Если хотите, можете добавить это слово в свои карточки — нажмите кнопку ниже."
                if reply:
                    reply = f"{reply}\n\n{invitation}"
                else:
                    reply = invitation
            await update.message.reply_text(
                reply,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=markup,
            )
            self._history.record(chat.id, user_message, reply)
            await self._increment_user_stats(chat.id, questions=1)


    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process a photo, combine caption instructions, and reply with analysis plus translation."""
        message = update.message
        if not message or not message.photo:
            return

        chat = update.effective_chat
        if chat is None:
            return

        self._pending_flashcard_terms.pop(chat.id, None)

        caption_text = (message.caption or "").strip()

        user = update.effective_user
        if user is not None:
            await self._store_user_profile(
                chat.id,
                getattr(user, "first_name", None),
                getattr(user, "last_name", None),
            )

        largest_photo = message.photo[-1]
        try:
            image_bytes = await download_photo_bytes(context.bot, largest_photo.file_id)
        except Exception:  # pragma: no cover - network errors
            LOGGER.exception("Failed to download photo for chat %s.", chat.id)
            await message.reply_text("Не удалось получить изображение. Попробуй отправить фото ещё раз.")
            return

        typing_task = asyncio.create_task(self._typing_indicator(chat.id, context))
        recognition_failed = False
        result: Optional[ImageTextResult] = None

        try:
            result = await extract_greek_text_from_image(
                self._client,
                self._vision_model,
                image_bytes,
                chat_id=chat.id,
            )
        except RuntimeError:
            recognition_failed = True
            LOGGER.exception("Image recognition failed for chat %s.", chat.id)
        finally:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task

        if recognition_failed:
            reply_text = "Не получилось распознать текст на этом фото. Попробуй ещё раз немного позже."
            history_entry = self._history.build_photo_history_entry(caption_text, None)
            await message.reply_text(reply_text)
            self._history.record(chat.id, history_entry, reply_text)
            await self._increment_user_stats(chat.id, questions=1)
            return

        if result is None:
            reply_text = "Похоже, на фото нет греческого текста. Пришли изображение, где текст именно на греческом."
            history_entry = self._history.build_photo_history_entry(caption_text, None)
            await message.reply_text(reply_text)
            self._history.record(chat.id, history_entry, reply_text)
            await self._increment_user_stats(chat.id, questions=1)
            return

        assistant_reply = ""
        caption_notice: Optional[str] = None
        conversation_prompt: Optional[str] = None

        if caption_text:
            conversation_prompt = build_photo_prompt(caption_text, result)
            caption_task = asyncio.create_task(self._typing_indicator(chat.id, context))
            try:
                assistant_reply = await self._generate_response(chat.id, conversation_prompt)
            except Exception:  # pragma: no cover - network issues
                LOGGER.exception("Failed to generate caption-aware response for chat %s.", chat.id)
                caption_notice = "Не удалось обработать запрос из подписи, но перевод текста приведён выше."
            finally:
                caption_task.cancel()
                with suppress(asyncio.CancelledError):
                    await caption_task

        if assistant_reply:
            assistant_reply = self._strip_add_card_prompts(assistant_reply).strip()

        common_lookup: Set[str] = self._get_common_word_lookup() or set()
        unique_words = collect_unique_image_words(
            result,
            normalize_term=self._normalize_greek_term,
            common_lookup=common_lookup,
            is_probable_proper_noun=self._is_probable_proper_noun,
        )
        selected_words = select_image_words_for_review(unique_words)
        selected_words, missing_words, words_checked = await analyze_image_words(
            chat.id,
            selected_words,
            can_check_flashcards=self._session_factory is not None and self._flashcard_workflow is not None,
            user_has_active_flashcard=self._user_has_active_flashcard,
        )
        flashcard_section = format_image_flashcard_section(
            selected_words,
            missing_words,
            checked=words_checked,
            normalize_term=self._normalize_greek_term,
            escape_html=self._escape_html,
        )

        reply_sections: List[str] = [format_image_result_message(result, self._escape_html)]
        if assistant_reply:
            reply_sections.append("<b>Ответ на запрос:</b>\n" + self._convert_markdown_to_html(assistant_reply))
        if caption_notice:
            reply_sections.append(self._escape_html(caption_notice))
        if flashcard_section:
            reply_sections.append(flashcard_section)

        reply_text = "\n\n".join(section for section in reply_sections if section).strip()

        reply_markup = self._build_take_card_markup()
        if words_checked and missing_words:
            pending_terms = [greek.strip() for greek, _ in missing_words if greek.strip()]
            if pending_terms:
                self._pending_flashcard_terms[chat.id] = pending_terms
                reply_markup = self._build_add_card_markup()
            else:
                self._pending_flashcard_terms.pop(chat.id, None)
        else:
            self._pending_flashcard_terms.pop(chat.id, None)

        await message.reply_text(
            reply_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )

        history_message = conversation_prompt or self._history.build_photo_history_entry(caption_text, result)
        self._history.record(chat.id, history_message, reply_text)
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

    async def _maybe_handle_flashcard_request(
        self,
        message: Optional[Message],
        chat_id: int,
        user_message: str,
    ) -> Optional[FlashcardWorkflowResult]:
        if message is None:
            return None

        if self._flashcard_workflow is None:
            return None

        context_payload = self._history.build_flashcard_context(chat_id)
        progress_message = None
        if self._flashcard_workflow.is_probable_request(user_message):
            with suppress(Exception):
                progress_message = await message.reply_text("Готовлю карточки...")

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

            await message.reply_text(
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

            await message.reply_text(
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
    def _build_add_card_markup() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("Добавить в карточки", callback_data="fc_add")]]
        )

    async def handle_add_flashcard(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        if query is None:
            return

        message = query.message
        if message is None or message.chat is None:
            await query.answer()
            return

        chat_id = message.chat.id
        pending_terms = self._pending_flashcard_terms.get(chat_id)
        if not pending_terms:
            await query.answer("Не нашёл подходящее слово для добавления.", show_alert=True)
            return

        if isinstance(pending_terms, str):
            terms = [pending_terms.strip()] if pending_terms.strip() else []
        else:
            terms = [
                term.strip()
                for term in pending_terms
                if isinstance(term, str) and term.strip()
            ]
        if not terms:
            self._pending_flashcard_terms.pop(chat_id, None)
            await query.answer("Подходящие слова не найдены.", show_alert=True)
            return

        await query.answer()
        self._pending_flashcard_terms.pop(chat_id, None)

        with suppress(Exception):
            await query.edit_message_reply_markup(reply_markup=None)

        for term in terms:
            synthetic_message = f"Добавь в карточки {term}"
            await self._maybe_handle_flashcard_request(message, chat_id, synthetic_message)

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

    @staticmethod
    def _convert_markdown_to_html(text: Optional[str]) -> str:
        """Convert minimal Markdown (bold/italic) to HTML while escaping other tags."""
        if not text:
            return ''

        escaped = GreekTeacherAgent._escape_html(text).replace("\r\n", "\n")

        def _replace(pattern: str, replacement: str, source: str) -> str:
            return re.sub(pattern, replacement, source, flags=re.DOTALL)

        # Bold (strong) **text** or __text__
        escaped = _replace(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = _replace(r"__(.+?)__", r"<b>\1</b>", escaped)
        # Italic *text* or _text_
        escaped = _replace(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
        escaped = _replace(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", escaped)

        return escaped

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

