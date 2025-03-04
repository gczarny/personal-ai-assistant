# src/database/models.py
from datetime import datetime, UTC

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Conversation(Base):
    """Model representing a conversation with a user."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    chat_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now(UTC))
    updated_at = Column(DateTime, default=datetime.now(UTC), onupdate=datetime.now(UTC))

    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """Model representing a single message in a conversation."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(20), nullable=False)  # 'system', 'user', or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now(UTC))

    conversation = relationship("Conversation", back_populates="messages")


class MediaMessage(Base):
    """Model representing a media message (image, voice, etc.)."""

    # TODO: Add support for media files
    __tablename__ = "media_messages"

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    media_type = Column(String(20), nullable=False)  # 'image', 'voice', etc.
    file_id = Column(String(100), nullable=True)
    caption = Column(Text, nullable=True)
    processed_text = Column(
        Text, nullable=True
    )  # For voice transcriptions or image descriptions

    message = relationship("Message", backref="media")
