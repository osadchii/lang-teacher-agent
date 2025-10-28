"""Utilities for maintaining per-chat conversation history."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Tuple


class PhotoSummary(Protocol):
    """Minimal interface required from OCR results when logging photo context."""

    greek_text: str
    translation: str


class MessageHistoryService:
    """Maintain rolling message history and build context payloads."""

    def __init__(self, history_size: int) -> None:
        self._history_size = history_size
        self._history: Dict[int, Deque[Tuple[str, str]]] = {}

    def get(self, chat_id: int) -> Deque[Tuple[str, str]]:
        """Return or create the message deque associated with a chat."""
        history = self._history.get(chat_id)
        if history is None:
            history = deque(maxlen=self._history_size)
            self._history[chat_id] = history
        return history

    def iter_history(self, chat_id: int) -> Iterable[Tuple[str, str]]:
        """Yield stored exchanges for the given chat in chronological order."""
        return tuple(self.get(chat_id))

    def record(self, chat_id: int, user_message: str, assistant_reply: str) -> None:
        """Append the latest user/assistant exchange to the chat history."""
        history = self.get(chat_id)
        history.append((user_message, assistant_reply))

    def build_messages(self, system_prompt: str, chat_id: int, user_message: str) -> List[Dict[str, str]]:
        """Compose a conversation payload compatible with the OpenAI Responses API."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for previous_user_message, previous_reply in self.iter_history(chat_id):
            messages.append({"role": "user", "content": previous_user_message})
            messages.append({"role": "assistant", "content": previous_reply})
        messages.append({"role": "user", "content": user_message})
        return messages

    def build_flashcard_context(self, chat_id: int) -> Optional[str]:
        """Describe the most recent exchange to help with flashcard extraction."""
        history = self.get(chat_id)
        if not history:
            return None
        previous_user, previous_reply = history[-1]
        lines: List[str] = []
        if previous_user:
            lines.append(
                f"Предыдущее сообщение пользователя: {previous_user.strip()}"
            )
        if previous_reply:
            lines.append(f"Ответ преподавателя: {previous_reply.strip()}")
        if not lines:
            return None
        lines.append("Добавь подходящую лексику из этого ответа в карточки ученика.")
        return "\n".join(lines)

    def build_photo_history_entry(
        self,
        caption: Optional[str],
        result: Optional[PhotoSummary],
    ) -> str:
        """Compose a history entry describing a processed photo."""
        lines: List[str] = ["[Фото]"]
        caption_text = (caption or "").strip()
        if caption_text:
            lines.append(f"Подпись: {caption_text}")

        if result is not None:
            lines.append("Распознанный текст:")
            lines.append(result.greek_text.strip())
            if result.translation:
                lines.append("Перевод:")
                lines.append(result.translation.strip())

        return "\n".join(lines).strip()

