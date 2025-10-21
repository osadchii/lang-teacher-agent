"""Utilities for working with OpenAI Responses API payloads."""

from __future__ import annotations

from typing import List


def extract_output_text(response: object) -> str:
    """Best-effort extraction of text from an OpenAI Responses result."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        collected: List[str] = []
        for item in output:
            maybe_text = getattr(getattr(item, "content", None), "text", None)
            if isinstance(maybe_text, list):
                for part in maybe_text:
                    if getattr(part, "type", None) == "output_text":
                        value = getattr(part, "value", None)
                        if value:
                            collected.append(str(value))
            elif isinstance(maybe_text, str):
                collected.append(maybe_text)
        if collected:
            return "\n".join(collected)

    return ""
