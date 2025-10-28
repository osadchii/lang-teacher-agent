"""Utilities for processing images inside the Greek teacher agent."""

from __future__ import annotations

import base64
import imghdr
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Awaitable, Callable, List, Optional, Sequence, Set, Tuple

from openai import AsyncOpenAI
from telegram import Bot

from src.bot.openai_utils import extract_output_text


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


EscapeHtmlFn = Callable[[Optional[str]], str]
NormalizeTermFn = Callable[[str], str]
IsProperNounFn = Callable[[str, Set[str]], bool]
UserHasFlashcardFn = Callable[[int, str], Awaitable[bool]]


async def download_photo_bytes(bot: Bot, file_id: str) -> bytes:
    """Download a Telegram photo by file_id and return its raw bytes."""
    telegram_file = await bot.get_file(file_id)
    buffer = BytesIO()
    await telegram_file.download_to_memory(out=buffer)
    return buffer.getvalue()


async def extract_greek_text_from_image(
    client: AsyncOpenAI,
    model: str,
    image_bytes: bytes,
    *,
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
        response = await client.responses.create(
            model=model,
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
    cleaned = _strip_code_fences(raw_text)
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
                    if stripped_entry and _contains_greek_characters(stripped_entry):
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

    if not greek_text or not _contains_greek_characters(greek_text):
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


def format_image_result_message(result: ImageTextResult, escape_html: EscapeHtmlFn) -> str:
    """Compose an HTML response summarizing OCR findings and translation."""
    lines: List[str] = [
        "<b>Найденный греческий текст:</b>",
        f"<pre>{escape_html(result.greek_text)}</pre>",
        "<b>Перевод:</b>",
        escape_html(result.translation or "Перевод нужно уточнить."),
    ]

    if result.words:
        lines.append("")
        lines.append("<b>Слова из текста:</b>")
        for greek_word, russian_word in result.words:
            if russian_word:
                lines.append(f"- {escape_html(greek_word)} - {escape_html(russian_word)}")
            else:
                lines.append(f"- {escape_html(greek_word)} - перевод нужно уточнить")

    if 0 < result.word_count <= 5:
        lines.append("")
        lines.append(
            "Если хочешь, помогу добавить эти слова в карточки (до 10 за раз) — просто напиши."
        )

    return "\n".join(lines).strip()


def build_photo_prompt(caption: str, result: ImageTextResult) -> str:
    """Prepare a user message for LLM when a caption accompanies a photo."""
    caption_text = caption.strip()
    lines: List[str] = []
    if caption_text:
        lines.append(f"Пользователь прислал фото с подписью: {caption_text}")
    else:
        lines.append("Пользователь прислал фото без подписи.")

    lines.append("")
    lines.append("Распознанный греческий текст с изображения:")
    lines.append(result.greek_text.strip())

    if result.translation:
        lines.append("")
        lines.append("Перевод распознанного текста на русский:")
        lines.append(result.translation.strip())

    lines.append("")
    lines.append(
        "Ответь на просьбу пользователя, используя этот материал. Если он просит проверить ошибки, "
        "выполнить задание или ответить на вопрос, проанализируй греческий текст и объясни результат по-русски."
    )
    return "\n".join(lines).strip()


def collect_unique_image_words(
    result: ImageTextResult,
    *,
    normalize_term: NormalizeTermFn,
    common_lookup: Set[str],
    is_probable_proper_noun: IsProperNounFn,
) -> List[Tuple[str, str]]:
    """Return unique Greek words extracted from OCR, preserving order."""
    unique: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for greek_word, translation_word in result.words:
        normalized = normalize_term(greek_word)
        if not normalized or normalized in seen:
            continue
        if is_probable_proper_noun(greek_word, common_lookup):
            continue
        seen.add(normalized)
        greek_clean = greek_word.strip()
        translation_clean = translation_word.strip() if translation_word else ""
        unique.append((greek_clean, translation_clean))
    return unique


def select_image_words_for_review(unique_words: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Select 1 or up to 3 key words from the OCR result depending on text length."""
    if not unique_words:
        return []

    limit = 1 if len(unique_words) <= 5 else 3
    return list(unique_words[:limit])


async def analyze_image_words(
    chat_id: int,
    selected_words: Sequence[Tuple[str, str]],
    *,
    can_check_flashcards: bool,
    user_has_active_flashcard: UserHasFlashcardFn,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], bool]:
    """
    Classify highlighted words by whether they already exist in the user's flashcards.

    Returns a tuple of (selected_words, missing_words, checked_flag).
    """
    selected_list = list(selected_words)
    if not selected_list:
        return [], [], False

    if not can_check_flashcards:
        return selected_list, [], False

    missing_words: List[Tuple[str, str]] = []
    for greek_word, translation_word in selected_list:
        if await user_has_active_flashcard(chat_id, greek_word):
            continue
        missing_words.append((greek_word, translation_word))

    return selected_list, missing_words, True


def format_image_flashcard_section(
    selected_words: Sequence[Tuple[str, str]],
    missing_words: Sequence[Tuple[str, str]],
    *,
    checked: bool,
    normalize_term: NormalizeTermFn,
    escape_html: EscapeHtmlFn,
) -> Optional[str]:
    """Format the section describing highlighted words and flashcard status."""
    if not selected_words:
        return None

    lines: List[str] = ["<b>Ключевые слова:</b>"]
    missing_lookup = (
        {normalize_term(greek): translation for greek, translation in missing_words}
        if checked
        else {}
    )

    for greek_word, translation_word in selected_words:
        normalized = normalize_term(greek_word)
        if translation_word:
            base_line = f"- {escape_html(greek_word)} - {escape_html(translation_word)}"
        else:
            base_line = f"- {escape_html(greek_word)}"

        if checked:
            if normalized in missing_lookup:
                base_line += " (нет в карточках)"
            else:
                base_line += " (уже есть в карточках)"

        lines.append(base_line)

    if checked and missing_words:
        lines.append("")
        missing_list = ", ".join(escape_html(greek) for greek, _ in missing_words)
        if missing_list:
            lines.append(f"Предлагаю добавить: {missing_list}.")
        lines.append("Отсутствующие слова можно добавить в карточки кнопкой ниже.")

    return "\n".join(lines).strip()


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


def _contains_greek_characters(text: str) -> bool:
    """Check if the provided text includes any Greek alphabet characters."""
    for char in text:
        code_point = ord(char)
        if 0x0370 <= code_point <= 0x03FF or 0x1F00 <= code_point <= 0x1FFF:
            return True
    return False
