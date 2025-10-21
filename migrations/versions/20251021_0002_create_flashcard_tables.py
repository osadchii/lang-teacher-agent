"""Create flashcard tables shared across users."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20251021_0002"
down_revision: Union[str, None] = "20251020_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "flashcards",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("target_text", sa.Text(), nullable=False),
        sa.Column("example", sa.Text(), nullable=True),
        sa.Column("source_lang", sa.String(length=32), nullable=True),
        sa.Column("target_lang", sa.String(length=32), nullable=True),
        sa.Column("tags", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            server_onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "source_text",
            "target_text",
            "source_lang",
            "target_lang",
            name="uq_flashcards_text_lang",
        ),
    )

    op.create_table(
        "user_flashcards",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("flashcard_id", sa.Integer(), nullable=False),
        sa.Column("easiness_factor", sa.Float(), server_default=sa.text("2.5"), nullable=False),
        sa.Column("interval", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("repetition", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "next_review_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_score", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            server_onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ("chat_id",),
            ("users.chat_id",),
            name="fk_user_flashcards_chat_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ("flashcard_id",),
            ("flashcards.id",),
            name="fk_user_flashcards_flashcard_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("chat_id", "flashcard_id", name="uq_user_flashcards_user_card"),
    )
    op.create_index(
        "ix_user_flashcards_chat_id_next_review_at",
        "user_flashcards",
        ("chat_id", "next_review_at"),
    )

    op.create_table(
        "flashcard_reviews",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("user_flashcard_id", sa.Integer(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column(
            "reviewed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ("user_flashcard_id",),
            ("user_flashcards.id",),
            name="fk_flashcard_reviews_user_flashcard_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_flashcard_reviews_user_flashcard_id",
        "flashcard_reviews",
        ("user_flashcard_id",),
    )


def downgrade() -> None:
    op.drop_index("ix_flashcard_reviews_user_flashcard_id", table_name="flashcard_reviews")
    op.drop_table("flashcard_reviews")
    op.drop_index("ix_user_flashcards_chat_id_next_review_at", table_name="user_flashcards")
    op.drop_table("user_flashcards")
    op.drop_table("flashcards")
