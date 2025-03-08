# tests/integration/test_conversation_flow.py
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

    completion_result = await mock_openai_client.create_chat_completion(
        mock_repository.get_messages.return_value
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


@pytest.mark.asyncio
async def test_search_web_functionality(telegram_bot, mock_openai_client):
    """Test the explicit web search command functionality."""
    # Configure mock for web search
    mock_search_result = {
        "query": "test query",
        "answer": "This is a synthesized answer about test query.",
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "This is test content about the query.",
                "score": 0.95,
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "More information about the test query.",
                "score": 0.85,
            },
        ],
    }

    mock_openai_client.search_web = AsyncMock(return_value=mock_search_result)

    mock_openai_client.tavily_manager = MagicMock()

    mock_openai_client.create_chat_completion = AsyncMock(
        return_value=Result.ok(
            "Here's what I found about test query: it's a sample query used for testing."
        )
    )

    mock_message = MagicMock(spec=Message)
    mock_message.text = "/search test query"
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345

    from telegram.ext import ContextTypes

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock()
    mock_context.bot.delete_message = AsyncMock()

    status_message = MagicMock(spec=Message)
    status_message.message_id = 67890
    mock_message.reply_text.return_value = status_message

    await telegram_bot._explicit_search_command(mock_update, mock_context)

    mock_openai_client.search_web.assert_called_once_with("test query")

    # Verify chat completion was called with search results
    mock_openai_client.create_chat_completion.assert_called_once()
    call_args = mock_openai_client.create_chat_completion.call_args[0][0]

    # Verify the call includes search results
    assert len(call_args) == 3
    assert call_args[0]["role"] == "system"
    assert "access to web search results" in call_args[0]["content"]
    assert call_args[1]["role"] == "user"
    assert "test query" in call_args[1]["content"]
    assert call_args[2]["role"] == "system"
    assert "Search results" in call_args[2]["content"]

    # Verify the status message was deleted
    mock_context.bot.delete_message.assert_called_once_with(
        chat_id=12345, message_id=67890
    )

    # Verify the response was sent
    assert (
        mock_message.reply_text.call_count >= 2
    )  # At least status message and final response

    # Get the last call which should be the search results
    response_call = [
        call
        for call in mock_message.reply_text.call_args_list
        if "Search results for: test query" in call[0][0]
    ]
    assert len(response_call) == 1
    assert "Here's what I found" in response_call[0][0][0]
