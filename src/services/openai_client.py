"""Helpers for configuring the OpenAI client."""

from openai import AsyncOpenAI


def build_openai_client(api_key: str) -> AsyncOpenAI:
    """Create a configured AsyncOpenAI client."""
    return AsyncOpenAI(api_key=api_key)
