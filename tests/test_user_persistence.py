import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import List, Tuple

import pytest

from src.db import run_migrations_if_needed
from src.db.users import upsert_user


class _StubSession:
    def __init__(self) -> None:
        self._records: dict[int, object] = {}
        self.flush_calls = 0

    async def get(self, model: object, chat_id: int) -> object:
        return self._records.get(chat_id)

    def add(self, user: object) -> None:
        self._records[getattr(user, "chat_id")] = user

    async def flush(self) -> None:
        self.flush_calls += 1


async def _exercise_user_upsert() -> None:
    session = _StubSession()

    created = await upsert_user(session, chat_id=101, first_name="Anna", last_name="Papadopoulos")
    assert session._records[101] is created
    assert created.first_name == "Anna"
    assert created.last_name == "Papadopoulos"
    assert created.created_at.tzinfo is not None
    assert created.updated_at == created.created_at
    assert session.flush_calls == 0

    updated = await upsert_user(session, chat_id=101, first_name="Annika", last_name="Papadopoulos")
    assert updated is created
    assert updated.first_name == "Annika"
    assert updated.last_name == "Papadopoulos"
    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    assert session.flush_calls == 1

    unchanged = await upsert_user(session, chat_id=101, first_name="Annika", last_name="Papadopoulos")
    assert unchanged is created
    assert session.flush_calls == 1  # no new flush when data is unchanged


def test_upsert_user_creates_and_updates_names() -> None:
    asyncio.run(_exercise_user_upsert())


def test_run_migrations_if_needed_invokes_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Tuple[object, str]] = []

    def fake_upgrade(config: object, target: str) -> None:
        calls.append((config, target))

    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("RUN_MIGRATIONS_ON_STARTUP", "true")
    monkeypatch.setattr("src.db.command.upgrade", fake_upgrade)

    run_migrations_if_needed()

    assert calls and calls[0][1] == "head"


def test_run_migrations_if_needed_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("RUN_MIGRATIONS_ON_STARTUP", "false")

    calls: deque[str] = deque()

    def fake_upgrade(_: object, target: str) -> None:
        calls.append(target)

    monkeypatch.setattr("src.db.command.upgrade", fake_upgrade)

    run_migrations_if_needed()

    assert not calls
