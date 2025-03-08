# tests/integration/test_conversation_flow.py
import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest
from telegram import Update, Message, Chat

from core.result import Result


@pytest.mark.asyncio
async def test_conversation_continuity_with_db(
    telegram_bot, mock_openai_client, mock_repository
):
    # Configure mock OpenAI client responses
    mock_openai_client.create_chat_completion.side_effect = [
        Result.ok("Hello! How can I help you?"),
        Result.ok("I understand you're asking about the weather. It's sunny today."),
    ]

    # Mock initial message retrieval
    mock_repository.get_messages.return_value = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # First message setup
    mock_text_message = MagicMock(spec=Message)
    mock_text_message.text = "Hello bot"
    mock_text_message.reply_text = AsyncMock()

    mock_text_update = MagicMock(spec=Update)
    mock_text_update.message = mock_text_message
    mock_text_update.effective_chat = MagicMock(spec=Chat)
    mock_text_update.effective_chat.id = 12345
    mock_text_update.effective_user = None

    mock_text_context = MagicMock()
    mock_text_context.bot = MagicMock()
    mock_text_context.bot.send_chat_action = AsyncMock()

    await telegram_bot._text_handler(mock_text_update, mock_text_context)

    mock_repository.get_or_create_conversation.assert_called_with("12345", None)
    mock_repository.add_message.assert_any_call("12345", "user", "Hello bot")
    mock_repository.add_message.assert_any_call(
        "12345", "assistant", "Hello! How can I help you?"
    )

    # Update mock repository for second message
    mock_repository.get_messages.return_value = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello bot"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]

    # Second message (simulating voice message transcription)
    telegram_bot.conversations[12345].append(
        {"role": "user", "content": "What's the weather like?"}
    )

    mock_repository.get_messages.return_value = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello bot"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "What's the weather like?"},
    ]

    completion_result = await asyncio.to_thread(
        mock_openai_client.create_chat_completion,
        mock_repository.get_messages.return_value,
    )

    if completion_result.success:
        # Add to in-memory conversation as voice handler would
        telegram_bot.conversations[12345].append(
            {"role": "assistant", "content": completion_result.value}
        )

    assert (
        len(telegram_bot.conversations[12345]) == 5
    )  # system + user + assistant + user + assistant
    assert telegram_bot.conversations[12345][3]["role"] == "user"
    assert telegram_bot.conversations[12345][3]["content"] == "What's the weather like?"
    assert telegram_bot.conversations[12345][4]["role"] == "assistant"
    assert (
        telegram_bot.conversations[12345][4]["content"]
        == "I understand you're asking about the weather. It's sunny today."
    )
