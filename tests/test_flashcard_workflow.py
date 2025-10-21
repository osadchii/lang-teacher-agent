from __future__ import annotations

import types

import pytest

from src.bot.flashcard_workflow import FlashcardWorkflow
from src.db.flashcards import get_next_flashcard_for_user


class _StubResponses:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0
        self.last_args: tuple | None = None
        self.last_kwargs: dict | None = None

    async def create(self, *args, **kwargs):
        self.calls += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return types.SimpleNamespace(output_text=self._payload)


class _StubClient:
    def __init__(self, payload: str) -> None:
        self.responses = _StubResponses(payload)


FLASHCARD_JSON = """
{
  "should_add": true,
  "flashcards": [
    {
      "source_text": "λόγος",
      "target_text": "слово",
      "example": "Ο λόγος είναι σημαντικός.",
      "example_translation": "Слово важно."
    }
  ]
}
"""


@pytest.mark.asyncio
async def test_workflow_persists_and_reuses_flashcards(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result = await workflow.handle(77, "Добавь слово λόγος в карточки")

    assert result.handled is True
    assert result.errors == []
    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.status == "created"
    assert summary.example.endswith("Слово важно.")
    assert client.responses.calls == 1

    async with session_factory() as session:
        stored = await get_next_flashcard_for_user(session, 77)
        assert stored is not None
        assert stored.flashcard.source_text == "λόγος"
        assert stored.flashcard.target_text == "слово"

    repeat_result = await workflow.handle(77, "Добавь это слово ещё раз в карточки")
    assert repeat_result.handled is True
    assert repeat_result.summaries[0].status == "existing"
    assert client.responses.calls == 2


@pytest.mark.asyncio
async def test_workflow_skips_irrelevant_messages(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    result = await workflow.handle(88, "Привет, как дела?")

    assert result.handled is False
    assert client.responses.calls == 0



@pytest.mark.asyncio
async def test_workflow_includes_context_for_extraction(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    context = (
        "Предыдущее сообщение пользователя: Как будет «больница»?\n"
        "Ответ преподавателя: По-гречески «больница» — το νοσοκομείο (to nosokomío)."
    )

    await workflow.handle(99, "Добавь в карточки", context=context)

    assert client.responses.calls == 1
    assert client.responses.last_kwargs is not None
    user_prompt = client.responses.last_kwargs["input"][1]["content"]
    assert "Добавь в карточки" in user_prompt
    assert "Контекст для создания карточек" in user_prompt
    assert "το νοσοκομείο" in user_prompt


def test_system_prompt_discourages_extra_variants(session_factory) -> None:
    client = _StubClient(FLASHCARD_JSON)
    workflow = FlashcardWorkflow(
        client=client,
        model="test-model",
        session_factory=session_factory,
        source_language="Greek",
        target_language="Russian",
    )

    prompt = workflow._system_prompt
    assert "Only produce flashcards for the exact terms explicitly requested" in prompt
    assert "Do not add extra grammatical variants" in prompt
