"""Tests for the short-term conversation memory maintained by the agent."""

from typing import Any

from src.main import GreekTeacherAgent, SYSTEM_PROMPT


class _StubResponses:
    async def create(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not used in tests
        raise AssertionError("Network calls are not expected in history tests.")


class _StubClient:
    def __init__(self) -> None:
        self.responses = _StubResponses()


def _build_agent(history_size: int = 5) -> GreekTeacherAgent:
    return GreekTeacherAgent(_StubClient(), "test-model", SYSTEM_PROMPT, history_size=history_size)


def test_history_is_trimmed_to_last_five_pairs() -> None:
    agent = _build_agent()
    chat_id = 101

    for index in range(6):
        agent._record_interaction(chat_id, f"user {index}", f"assistant {index}")

    messages = agent._build_messages(chat_id, "latest question")

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    expected_pairs = [(f"user {i}", f"assistant {i}") for i in range(1, 6)]
    for pair_index, (user_text, assistant_text) in enumerate(expected_pairs, start=1):
        user_message = messages[pair_index * 2 - 1]
        assistant_message = messages[pair_index * 2]
        assert user_message == {"role": "user", "content": user_text}
        assert assistant_message == {"role": "assistant", "content": assistant_text}

    assert messages[-1] == {"role": "user", "content": "latest question"}


def test_histories_are_isolated_by_chat_id() -> None:
    agent = _build_agent(history_size=3)
    first_chat = 1
    second_chat = 2

    agent._record_interaction(first_chat, "user first", "assistant first")
    agent._record_interaction(second_chat, "user second", "assistant second")

    first_messages = agent._build_messages(first_chat, "follow up")
    second_messages = agent._build_messages(second_chat, "new question")

    assert {message["content"] for message in first_messages if message["role"] == "assistant"} == {
        "assistant first"
    }
    assert {message["content"] for message in second_messages if message["role"] == "assistant"} == {
        "assistant second"
    }


def test_flashcard_context_uses_last_exchange() -> None:
    agent = _build_agent()
    chat_id = 303

    agent._record_interaction(
        chat_id,
        "Как будет «больница»?",
        "По-гречески «больница» — το νοσοκομείο (to nosokomío).",
    )

    context = agent._build_flashcard_context(chat_id)

    assert context is not None
    assert "Как будет «больница»?" in context
    assert "το νοσοκομείο" in context

