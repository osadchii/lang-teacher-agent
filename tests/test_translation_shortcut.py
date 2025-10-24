from __future__ import annotations

import types

import pytest
from telegram import InlineKeyboardMarkup

from src.main import GreekTeacherAgent, SYSTEM_PROMPT
from src.db.flashcards import FlashcardPayload, ensure_user_flashcard, get_or_create_flashcard
from src.db.users import upsert_user


class _StubResponses:
    async def create(self, *args, **kwargs):  # pragma: no cover - network calls not expected in tests
        raise AssertionError("Unexpected OpenAI call during tests.")


class _StubClient:
    def __init__(self) -> None:
        self.responses = _StubResponses()


@pytest.mark.asyncio
async def test_resolve_reply_markup_offers_add_button_for_new_translation(session_factory) -> None:
    agent = GreekTeacherAgent(
        _StubClient(),
        "test-model",
        SYSTEM_PROMPT,
        session_factory=session_factory,
    )

    chat_id = 301
    user_message = "Как по-гречески «больница»?"
    assistant_reply = (
        "**Слово:** больница\n"
        "**По-гречески:** το νοσοκομείο (to nosokomío)\n"
        "**Пример:** Στο νοσοκομείο δουλεύει ο γιατρός."
    )

    markup, notice = await agent._resolve_reply_markup(chat_id, user_message, assistant_reply)

    assert isinstance(markup, InlineKeyboardMarkup)
    assert markup.inline_keyboard[0][0].text == "Добавить в карточки"
    assert notice is None
    assert agent._pending_flashcard_terms[chat_id] == "το νοσοκομείο"


@pytest.mark.asyncio
async def test_resolve_reply_markup_defaults_when_flashcard_exists(session_factory) -> None:
    agent = GreekTeacherAgent(
        _StubClient(),
        "test-model",
        SYSTEM_PROMPT,
        session_factory=session_factory,
    )

    chat_id = 302
    greek_term = "το νοσοκομείο"

    async with session_factory() as session:
        async with session.begin():
            await upsert_user(session, chat_id, "Иван", None)
            flashcard, _ = await get_or_create_flashcard(
                session,
                FlashcardPayload(
                    source_text=greek_term,
                    target_text="больница",
                    source_lang="Greek",
                    target_lang="Russian",
                ),
            )
            await ensure_user_flashcard(session, chat_id, flashcard)

    markup, notice = await agent._resolve_reply_markup(
        chat_id,
        "Как по-гречески «больница»?",
        "**По-гречески:** το νοσοκομείο (to nosokomío)",
    )

    assert isinstance(markup, InlineKeyboardMarkup)
    assert markup.inline_keyboard[0][0].text == "Взять карточку"
    assert chat_id not in agent._pending_flashcard_terms
    assert notice == "Это слово уже есть в твоих карточках — потренируйся, чтобы закрепить значение!"


def test_strip_add_card_prompts_removes_add_lines() -> None:
    agent = GreekTeacherAgent(_StubClient(), "test-model", SYSTEM_PROMPT)
    original = (
        "Дом по-гречески — το σπίτι.\n"
        "Хочешь, я добавлю это слово в твои карточки?\n"
        "Могу сохранить карточку с этим словом, чтобы тебе было легче.\n"
        "Продолжай тренироваться!"
    )
    cleaned = agent._strip_add_card_prompts(original)
    assert "добавлю" not in cleaned.casefold()
    assert "сохран" not in cleaned.casefold()
    assert "Дом по-гречески — το σπίτι." in cleaned
    assert "Продолжай тренироваться!" in cleaned


class _StubMessage:
    def __init__(self, chat_id: int) -> None:
        self.chat = types.SimpleNamespace(id=chat_id)


class _StubCallbackQuery:
    def __init__(self, message: _StubMessage) -> None:
        self.message = message
        self.answered = False
        self.alert_message: str | None = None
        self.markup_cleared = False

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answered = True
        if show_alert:
            self.alert_message = text

    async def edit_message_reply_markup(self, reply_markup=None) -> None:
        self.markup_cleared = reply_markup is None


@pytest.mark.asyncio
async def test_handle_add_flashcard_uses_pending_term(session_factory) -> None:
    agent = GreekTeacherAgent(
        _StubClient(),
        "test-model",
        SYSTEM_PROMPT,
        session_factory=session_factory,
    )

    chat_id = 303
    pending_term = "το νοσοκομείο"
    agent._pending_flashcard_terms[chat_id] = pending_term

    captured: dict[str, object] = {}

    async def fake_handle(message, chat_id_value, user_message):
        captured["message"] = message
        captured["chat_id"] = chat_id_value
        captured["user_message"] = user_message

    agent._maybe_handle_flashcard_request = fake_handle  # type: ignore[assignment]

    message = _StubMessage(chat_id)
    query = _StubCallbackQuery(message)
    update = types.SimpleNamespace(callback_query=query)

    await agent.handle_add_flashcard(update, None)

    assert captured["chat_id"] == chat_id
    assert captured["message"] is message
    assert captured["user_message"] == f"Добавь в карточки {pending_term}"
    assert query.answered is True
    assert query.alert_message is None
    assert query.markup_cleared is True
    assert chat_id not in agent._pending_flashcard_terms
