import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


LOGGER = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""


class User(Base):
    """Represents a Telegram user interacting with the bot."""

    __tablename__ = "users"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    questions_asked: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    flashcards_added: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    flashcards_reviewed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    flashcards_mastered: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        server_onupdate=func.now(),
    )
    flashcards: Mapped[list["UserFlashcard"]] = relationship(
        "UserFlashcard",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Flashcard(Base):
    """Shared flashcard definition that can be used by multiple users."""

    __tablename__ = "flashcards"
    __table_args__ = (
        UniqueConstraint(
            "source_text",
            "target_text",
            "source_lang",
            "target_lang",
            name="uq_flashcards_text_lang",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    example: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_lang: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    target_lang: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        server_onupdate=func.now(),
    )
    user_links: Mapped[list["UserFlashcard"]] = relationship(
        "UserFlashcard",
        back_populates="flashcard",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class UserFlashcard(Base):
    """Flashcard settings tailored to a specific user."""

    __tablename__ = "user_flashcards"
    __table_args__ = (
        UniqueConstraint("chat_id", "flashcard_id", name="uq_user_flashcards_user_card"),
        Index("ix_user_flashcards_chat_id_next_review_at", "chat_id", "next_review_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.chat_id", ondelete="CASCADE"), nullable=False
    )
    flashcard_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("flashcards.id", ondelete="CASCADE"), nullable=False
    )
    easiness_factor: Mapped[float] = mapped_column(Float, nullable=False, default=2.5)
    interval: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    repetition: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_review_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        server_onupdate=func.now(),
    )
    user: Mapped["User"] = relationship("User", back_populates="flashcards")
    flashcard: Mapped["Flashcard"] = relationship("Flashcard", back_populates="user_links")
    reviews: Mapped[list["FlashcardReview"]] = relationship(
        "FlashcardReview",
        back_populates="user_flashcard",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class FlashcardReview(Base):
    """History of a user's spaced-repetition reviews for a flashcard."""

    __tablename__ = "flashcard_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_flashcard_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user_flashcards.id", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    user_flashcard: Mapped["UserFlashcard"] = relationship("UserFlashcard", back_populates="reviews")


def _expand_database_url(raw_url: str) -> str:
    """Expand environment variables inside the configured database URL."""
    return os.path.expandvars(raw_url)


def get_database_url() -> str:
    """Return the configured database URL or raise if missing."""
    raw_url = os.getenv("DATABASE_URL")
    if not raw_url:
        raise RuntimeError("DATABASE_URL environment variable is required to connect to the database.")
    return _expand_database_url(raw_url)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Create (and cache) the async engine for the application's database."""
    echo = os.getenv("SQLALCHEMY_ECHO", "false").lower() in {"1", "true", "yes"}
    return create_async_engine(get_database_url(), echo=echo)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a cached async session factory bound to the engine."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


def should_run_migrations() -> bool:
    """Determine whether migrations should be executed during startup."""
    flag = os.getenv("RUN_MIGRATIONS_ON_STARTUP", "true").lower()
    return flag in {"1", "true", "yes", "on"}


def _build_alembic_config() -> Config:
    project_root = Path(__file__).resolve().parents[2]
    alembic_cfg = Config(str(project_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(project_root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", get_database_url())
    return alembic_cfg


def run_migrations(target: str = "head") -> None:
    """Run Alembic migrations up to the specified target revision."""
    command.upgrade(_build_alembic_config(), target)


def run_migrations_if_needed(target: str = "head") -> None:
    """Run migrations when the startup flag is enabled."""
    if not should_run_migrations():
        LOGGER.info("Skipping migrations because RUN_MIGRATIONS_ON_STARTUP is disabled.")
        return

    LOGGER.info("Applying database migrations up to %s.", target)
    run_migrations(target)
    LOGGER.info("Database schema is up to date.")
