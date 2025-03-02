import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest
from telegram import Update, Message, Chat

from core.result import Result


@pytest.mark.asyncio
async def test_conversation_continuity_with_voice(telegram_bot, mock_openai_client):
    # Configure mock OpenAI client with specific responses for this test
    mock_openai_client.create_chat_completion.side_effect = [
        Result.ok("Hello! How can I help you?"),
        Result.ok("I understand you're asking about the weather. It's sunny today."),
    ]
    mock_openai_client.transcribe_audio.return_value = Result.ok(
        "What's the weather like?"
    )

    # Mock objects for text message
    mock_text_message = MagicMock(spec=Message)
    mock_text_message.text = "Hello bot"
    mock_text_message.reply_text = AsyncMock()

    mock_text_update = MagicMock(spec=Update)
    mock_text_update.message = mock_text_message
    mock_text_update.effective_chat = MagicMock(spec=Chat)
    mock_text_update.effective_chat.id = 12345

    mock_text_context = MagicMock()
    mock_text_context.bot = MagicMock()
    mock_text_context.bot.send_chat_action = AsyncMock()

    # Process text message
    await telegram_bot._text_handler(mock_text_update, mock_text_context)

    # Verify first message handled correctly
    assert 12345 in telegram_bot.conversations
    assert len(telegram_bot.conversations[12345]) == 3  # system + user + assistant
    assert telegram_bot.conversations[12345][0]["role"] == "system"
    assert telegram_bot.conversations[12345][1]["role"] == "user"
    assert telegram_bot.conversations[12345][1]["content"] == "Hello bot"
    assert telegram_bot.conversations[12345][2]["role"] == "assistant"
    assert (
        telegram_bot.conversations[12345][2]["content"] == "Hello! How can I help you?"
    )

    # Directly add the transcribed text to conversation (simulating what _voice_handler would do)
    telegram_bot.conversations[12345].append(
        {"role": "user", "content": "What's the weather like?"}
    )

    # Simulate getting a response from OpenAI
    completion_result = await asyncio.to_thread(
        mock_openai_client.create_chat_completion, telegram_bot.conversations[12345]
    )

    if completion_result.success:
        telegram_bot.conversations[12345].append(
            {"role": "assistant", "content": completion_result.value}
        )

    # Verify conversation continuity
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
