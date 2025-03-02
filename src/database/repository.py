# src/database/repository.py
from typing import List, Dict, Optional, Any

from sqlalchemy.orm import Session

from .models import Conversation, Message


class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create_conversation(
        self, chat_id: str, user_id: Optional[str] = None
    ) -> Conversation:
        """Get an existing conversation or create a new one."""
        conversation = (
            self.session.query(Conversation).filter_by(chat_id=chat_id).first()
        )
        if not conversation:
            conversation = Conversation(chat_id=chat_id, user_id=user_id)
            self.session.add(conversation)
            self.session.commit()
            self.session.refresh(conversation)
        elif user_id and conversation.user_id != user_id:
            conversation.user_id = user_id
            self.session.commit()
            self.session.refresh(conversation)
        return conversation

    def add_message(self, chat_id: str, role: str, content: str) -> Message:
        """Add a new message to the conversation."""
        conversation = self.get_or_create_conversation(chat_id)
        message = Message(conversation_id=conversation.id, role=role, content=content)
        self.session.add(message)
        self.session.commit()
        return message

    def get_messages(
        self, chat_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation messages in OpenAI format."""
        conversation = self.get_or_create_conversation(chat_id)
        query = (
            self.session.query(Message)
            .filter_by(conversation_id=conversation.id)
            .order_by(Message.timestamp)
        )

        if limit:
            messages = query.limit(limit).all()
        else:
            messages = query.all()

        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear_conversation(self, chat_id: str) -> None:
        """Delete all messages in a conversation but keep the conversation record."""
        conversation = self.get_or_create_conversation(chat_id)
        self.session.query(Message).filter_by(conversation_id=conversation.id).delete()
        self.session.commit()
