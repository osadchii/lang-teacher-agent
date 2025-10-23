"""Workflow for extracting, generating, and storing flashcards from user messages."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.bot.openai_utils import extract_output_text
from src.db.flashcards import (
    FlashcardPayload,
    ensure_user_flashcard,
    get_or_create_flashcard,
    get_user_flashcard_by_source_text,
)


LOGGER = logging.getLogger(__name__)


_TRIGGER_KEYWORDS = (
    "добав",
    "карточ",
    "flashcard",
    "card",
    "memor",
    "learn",
    "слово",
    "флэш",
    "повтори",
    "закрепи",
)


@dataclass(slots=True)
class GeneratedFlashcard:
    """Card information extracted from an OpenAI response."""

    source_text: str
    target_text: str
    example: Optional[str] = None
    example_translation: Optional[str] = None

    def to_payload(
        self,
        source_lang: Optional[str],
        target_lang: Optional[str],
    ) -> FlashcardPayload:
        """Convert to a persistence payload with combined example text."""
        example_text: Optional[str]
        if self.example and self.example_translation:
            example_text = f"{self.example.strip()} — {self.example_translation.strip()}"
        elif self.example:
            example_text = self.example.strip()
        elif self.example_translation:
            example_text = self.example_translation.strip()
        else:
            example_text = None
        return FlashcardPayload(
            source_text=self.source_text,
            target_text=self.target_text,
            example=example_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )


@dataclass(slots=True)
class FlashcardExtractionResult:
    """Outcome of attempting to extract flashcards from a message."""

    should_add: bool
    flashcards: List[GeneratedFlashcard]
    reason: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass(slots=True)
class FlashcardSummary:
    """Summary of stored flashcards for acknowledgement."""

    source_text: str
    target_text: str
    example: Optional[str]
    status: str  # "created", "reactivated", "existing"
    user_flashcard_id: Optional[int] = None


@dataclass(slots=True)
class FlashcardWorkflowResult:
    """Result of running the flashcard workflow on a message."""

    handled: bool
    summaries: List[FlashcardSummary]
    errors: List[str]
    reason: Optional[str] = None

    @property
    def has_new_cards(self) -> bool:
        return any(summary.status in {"created", "reactivated"} for summary in self.summaries)


class FlashcardWorkflow:
    """Coordinates LLM extraction and database persistence for flashcards."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        session_factory: Optional[async_sessionmaker[AsyncSession]],
        source_language: str,
        target_language: str,
        max_cards_per_message: int = 10,
    ) -> None:
        self._client = client
        self._model = model
        self._session_factory = session_factory
        self._source_language = source_language
        self._target_language = target_language
        self._max_cards_per_message = max_cards_per_message
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return (
            "You help manage vocabulary flashcards for students learning {source_language}. "
            "Determine whether the user's message describes words or phrases that should be turned into spaced-repetition flashcards. "
            "If the message does not request flashcards, respond with JSON {{\"should_add\": false, \"flashcards\": [], \"reason\": \"<short reason>\"}}. "
            "If the user wants new flashcards, respond with JSON describing them. "
            "Return at most {max_cards} flashcards. Each flashcard must include: "
            "\"source_text\" (the term in {source_language}), "
            "\"target_text\" (a natural translation into {target_language}), "
            "\"example\" (a short sentence in {source_language}), and "
            "\"example_translation\" (the same sentence translated into {target_language}). "
            "Only produce flashcards for the exact terms explicitly requested by the user. "
            "Do not add extra grammatical variants such as plurals, cases, or other inflections unless the user clearly asks for them. "
            "When the term is a noun in {source_language}, prepend the correct definite article (such as 'ο', 'η', or 'το') to the source_text so the learner can see its grammatical gender. "
            "Always produce valid JSON without commentary, Markdown, or code fences."
        ).format(
            source_language=self._source_language,
            target_language=self._target_language,
            max_cards=self._max_cards_per_message,
        )

    def is_probable_request(self, message: str) -> bool:
        """Heuristic helper to detect if a message might contain flashcard content."""
        return self._looks_like_flashcard_request(message)

    @staticmethod
    def _looks_like_flashcard_request(message: str) -> bool:
        normalized = message.strip().lower()
        if len(normalized) < 5:
            return False
        return any(keyword in normalized for keyword in _TRIGGER_KEYWORDS)

    async def handle(
        self,
        chat_id: int,
        message: str,
        context: Optional[str] = None,
    ) -> FlashcardWorkflowResult:
        """Attempt to create flashcards from a user message."""
        if self._session_factory is None:
            return FlashcardWorkflowResult(False, [], ["Database session factory is not configured."])

        if not self._looks_like_flashcard_request(message):
            return FlashcardWorkflowResult(False, [], [])

        try:
            extraction_message = self._combine_message_and_context(message, context)
            extraction = await self._extract_flashcards(extraction_message)
        except Exception:
            LOGGER.exception("Flashcard extraction failed for chat %s.", chat_id)
            return FlashcardWorkflowResult(
                handled=False,
                summaries=[],
                errors=["Не удалось обработать карточки. Попробуйте сформулировать запрос иначе."],
            )

        if not extraction.should_add or not extraction.flashcards:
            return FlashcardWorkflowResult(
                handled=False,
                summaries=[],
                errors=extraction.errors,
                reason=extraction.reason,
            )

        summaries: List[FlashcardSummary] = []

        async with self._session_factory() as session:
            async with session.begin():
                for card in extraction.flashcards[: self._max_cards_per_message]:
                    summary = await self._store_flashcard(session, chat_id, card)
                    if summary:
                        summaries.append(summary)

        return FlashcardWorkflowResult(
            handled=True,
            summaries=summaries,
            errors=extraction.errors,
            reason=extraction.reason,
        )

    @staticmethod
    def _combine_message_and_context(
        message: str,
        context: Optional[str],
    ) -> str:
        if not context:
            return message

        message_block = message.strip()
        context_block = context.strip()
        if not message_block:
            return context_block

        return f"{message_block}\n\nКонтекст для создания карточек:\n{context_block}"

    async def _extract_flashcards(self, message: str) -> FlashcardExtractionResult:
        response = await self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": message},
            ],
        )
        raw_text = extract_output_text(response).strip()
        cleaned = self._strip_code_fences(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse flashcard extraction response: %s", cleaned)
            raise RuntimeError("Flashcard extraction returned malformed JSON.") from exc

        should_add = bool(payload.get("should_add"))
        reason = payload.get("reason")
        raw_cards = payload.get("flashcards") or []

        cards: List[GeneratedFlashcard] = []
        errors: List[str] = []
        for index, raw_card in enumerate(raw_cards, start=1):
            converted = self._convert_raw_card(raw_card, index)
            if isinstance(converted, GeneratedFlashcard):
                cards.append(converted)
            else:
                errors.append(converted)

        return FlashcardExtractionResult(
            should_add=should_add,
            flashcards=cards,
            reason=reason,
            errors=errors,
        )

    @staticmethod
    def _strip_code_fences(response_text: str) -> str:
        fenced = response_text.strip()
        if fenced.startswith("```") and fenced.endswith("```"):
            return fenced.split("\n", 1)[-1].rsplit("\n", 1)[0]
        return fenced

    def _convert_raw_card(self, raw_card: object, index: int) -> GeneratedFlashcard | str:
        if not isinstance(raw_card, dict):
            return f"Карточка #{index} имеет неверный формат."

        source_text = self._sanitize_text(raw_card.get("source_text"))
        target_text = self._sanitize_text(raw_card.get("target_text"))
        example = self._sanitize_text(raw_card.get("example"))
        example_translation = self._sanitize_text(raw_card.get("example_translation"))

        missing: List[str] = []
        if not source_text:
            missing.append("source_text")
        if not target_text:
            missing.append("target_text")
        if missing:
            return f"Карточка #{index} пропускает поля: {', '.join(missing)}."

        return GeneratedFlashcard(
            source_text=source_text,
            target_text=target_text,
            example=example,
            example_translation=example_translation,
        )

    @staticmethod
    def _sanitize_text(value: object) -> Optional[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    async def _store_flashcard(
        self,
        session: AsyncSession,
        chat_id: int,
        card: GeneratedFlashcard,
    ) -> Optional[FlashcardSummary]:
        now = datetime.now(timezone.utc)
        payload = card.to_payload(self._source_language, self._target_language).normalized()

        # Check if user already has a card with this Greek word
        existing_user_card = await get_user_flashcard_by_source_text(
            session, chat_id, payload.source_text, payload.source_lang
        )
        if existing_user_card:
            # User already has this Greek word - return existing card info
            return FlashcardSummary(
                source_text=existing_user_card.flashcard.source_text,
                target_text=existing_user_card.flashcard.target_text,
                example=existing_user_card.flashcard.example,
                status="existing",
                user_flashcard_id=existing_user_card.id,
            )

        # User doesn't have this word yet - proceed with normal flow
        flashcard, _ = await get_or_create_flashcard(session, payload)
        user_flashcard, created_user_link, reactivated = await ensure_user_flashcard(
            session, chat_id, flashcard, now=now
        )

        if not user_flashcard:
            return None

        if created_user_link:
            status = "created"
        elif reactivated:
            status = "reactivated"
        else:
            status = "existing"

        return FlashcardSummary(
            source_text=payload.source_text,
            target_text=payload.target_text,
            example=payload.example,
            status=status,
            user_flashcard_id=user_flashcard.id,
        )
