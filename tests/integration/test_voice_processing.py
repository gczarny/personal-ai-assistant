# tests/integration/test_voice_processing.py
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from telegram import Update, Voice, Message, Chat, File
from telegram.ext import ContextTypes

from core.result import Result


@pytest.mark.asyncio
async def test_voice_message_processing_flow(
    telegram_bot, mock_openai_client, sample_ogg
):
    # Configure mock OpenAI client with specific responses for this test
    mock_openai_client.transcribe_audio.return_value = Result.ok(
        "What is the weather like today?"
    )
    mock_openai_client.create_chat_completion.return_value = Result.ok(
        "The weather is sunny today."
    )

    # Create mock update and context objects
    mock_voice = MagicMock(spec=Voice)
    mock_voice.file_id = "test_file_id"
    mock_voice.duration = 3

    mock_message = MagicMock(spec=Message)
    mock_message.voice = mock_voice
    mock_message.reply_text = AsyncMock()

    mock_chat = MagicMock(spec=Chat)
    mock_chat.id = 12345

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = mock_chat

    # Read the sample voice file content
    with open(sample_ogg, "rb") as f:
        sample_voice_content = f.read()

    # Mock the file object that Telegram would return
    mock_file = AsyncMock(spec=File)
    mock_file.file_id = "test_file_id"
    mock_file.file_path = sample_ogg

    # Mock download_to_drive to actually write the file
    mock_file.download_to_drive = AsyncMock(
        side_effect=lambda path: open(path, "wb").write(sample_voice_content)
    )

    mock_bot = MagicMock()
    mock_bot.get_file = AsyncMock(return_value=mock_file)
    mock_bot.send_chat_action = AsyncMock()

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = mock_bot

    # Patch necessary functions to avoid actual processing
    with patch("clients.telegram_bot.AudioSegment") as mock_audio_segment, patch(
        "os.remove"
    ) as mock_remove, patch("os.path.exists", return_value=True):
        # Configure AudioSegment mock
        mock_audio = MagicMock()
        mock_audio.export = MagicMock()
        mock_audio_segment.from_ogg.return_value = mock_audio

        await telegram_bot._voice_handler(mock_update, mock_context)

        # Verify the complete flow
        assert mock_message.reply_text.call_count >= 3
        mock_bot.send_chat_action.assert_called()
        mock_bot.get_file.assert_called_once_with(mock_voice.file_id)
        mock_file.download_to_drive.assert_called_once()

        # Verify OpenAI client usage
        mock_openai_client.transcribe_audio.assert_called_once()
        mock_openai_client.create_chat_completion.assert_called_once()

        # Verify conversation state
        assert 12345 in telegram_bot.conversations
        assert len(telegram_bot.conversations[12345]) >= 2

        user_messages = [
            msg for msg in telegram_bot.conversations[12345] if msg["role"] == "user"
        ]
        assistant_messages = [
            msg
            for msg in telegram_bot.conversations[12345]
            if msg["role"] == "assistant"
        ]

        assert any(
            msg["content"] == "What is the weather like today?" for msg in user_messages
        )
        assert any(
            msg["content"] == "The weather is sunny today."
            for msg in assistant_messages
        )

        assert mock_remove.call_count >= 1
