# tests/unit/test_telegram_utils.py
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from telegram import Update, Message, Chat, Voice, File, PhotoSize
from telegram.ext import ContextTypes

from core.result import Result


@pytest.mark.asyncio
async def test_start_command(telegram_bot):
    # Mock dependencies
    mock_update = MagicMock(spec=Update)
    mock_update.effective_chat.id = 123456

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.send_message = AsyncMock()

    await telegram_bot._start_command(mock_update, mock_context)

    mock_context.bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_clear_command(telegram_bot, mock_repository):
    """Test the clear command functionality."""
    # Mock dependencies
    mock_message = MagicMock(spec=Message)
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 123456

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    # Set up existing in-memory conversation
    telegram_bot.conversations[123456] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"},
    ]

    # Execute the command
    await telegram_bot._clear_command(mock_update, mock_context)

    # Verify the repository interaction
    mock_repository.clear_conversation.assert_called_once_with("123456")

    # Verify in-memory conversation was reset
    assert 123456 in telegram_bot.conversations
    assert len(telegram_bot.conversations[123456]) == 1
    assert telegram_bot.conversations[123456][0]["role"] == "system"

    # Verify user was notified
    mock_message.reply_text.assert_called_once()
    assert "cleared" in mock_message.reply_text.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_text_handler_success(telegram_bot, mock_openai_client, mock_repository):
    # cxonfigure mock OpenAI client
    mock_openai_client.create_chat_completion.return_value = Result.ok(
        "This is a test response"
    )

    # onfigure mock repository
    test_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    mock_repository.get_messages.return_value = test_messages

    # Build a fake Update
    mock_user = MagicMock()
    mock_user.username = "test_user"

    mock_message = MagicMock(spec=Message)
    mock_message.text = "Hello bot"
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 123456
    mock_update.effective_user = mock_user

    # Build a fake context
    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot.send_chat_action = AsyncMock()

    # Call the bot handler
    await telegram_bot._text_handler(mock_update, mock_context)

    # 6) Assert your repository calls
    mock_repository.get_or_create_conversation.assert_called_with("123456", "test_user")
    mock_repository.add_message.assert_any_call("123456", "user", "Hello bot")
    mock_repository.add_message.assert_any_call(
        "123456", "assistant", "This is a test response"
    )

    # Check in-memory conversation
    assert 123456 in telegram_bot.conversations
    assert len(telegram_bot.conversations[123456]) == 3  # system + user + assistant
    assert telegram_bot.conversations[123456][0]["role"] == "system"
    assert telegram_bot.conversations[123456][1]["role"] == "user"
    assert telegram_bot.conversations[123456][1]["content"] == "Hello bot"
    assert telegram_bot.conversations[123456][2]["role"] == "assistant"
    assert telegram_bot.conversations[123456][2]["content"] == "This is a test response"

    # Check final user reply
    mock_message.reply_text.assert_called_once_with("This is a test response")


@pytest.mark.asyncio
async def test_process_image(telegram_bot, mock_openai_client):
    # Configure mock OpenAI client
    mock_openai_client.create_chat_completion.return_value = Result.ok(
        "This is a test image description"
    )

    # Create test image data
    image_data = b"test_image_data"
    caption = "Describe this image"
    file_path = "test_image.jpg"

    result = await telegram_bot._process_image(image_data, caption, file_path)

    assert result == "This is a test image description"

    # Verify the OpenAI client was called correctly
    mock_openai_client.create_chat_completion.assert_called_once()
    call_args = mock_openai_client.create_chat_completion.call_args[0][0]
    assert call_args[0]["role"] == "user"
    assert len(call_args[0]["content"]) == 2
    assert call_args[0]["content"][0]["type"] == "text"
    assert call_args[0]["content"][0]["text"] == caption
    assert call_args[0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_image_handler(telegram_bot):
    # Create a mock for process_image
    telegram_bot._process_image = AsyncMock(return_value="Test image description")

    # Mock dependencies
    mock_photo = MagicMock(spec=PhotoSize)
    mock_photo.file_id = "test_photo_id"

    mock_message = MagicMock(spec=Message)
    mock_message.photo = [mock_photo]
    mock_message.caption = "Describe this"
    mock_message.reply_text = AsyncMock()

    mock_file = MagicMock(spec=File)
    mock_file.file_path = "test.jpg"
    mock_file.download_to_memory = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 123456

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)
    mock_context.bot.send_chat_action = AsyncMock()

    # Create a BytesIO object that will be used to simulate file download
    file_data = BytesIO(b"test image data")
    file_data.seek(0)

    # Configure download_to_memory to write to the file_data
    async def mock_download_to_memory(buffer):
        buffer.write(b"test image data")
        buffer.seek(0)

    mock_file.download_to_memory.side_effect = mock_download_to_memory

    await telegram_bot._image_handler(mock_update, mock_context)

    telegram_bot._process_image.assert_called_once()
    mock_message.reply_text.assert_called_once_with("Test image description")


@pytest.mark.asyncio
async def test_voice_handler(telegram_bot, mock_openai_client, sample_ogg):
    # Configure mock OpenAI client
    mock_openai_client.transcribe_audio.return_value = Result.ok(
        "This is a test transcription"
    )
    mock_openai_client.create_chat_completion.return_value = Result.ok(
        "This is a response to your voice message"
    )

    # Mock dependencies
    mock_voice = MagicMock(spec=Voice)
    mock_voice.file_id = "test_voice_id"
    mock_voice.duration = 5

    mock_message = MagicMock(spec=Message)
    mock_message.voice = mock_voice
    mock_message.reply_text = AsyncMock()

    mock_file = MagicMock(spec=File)
    mock_file.file_path = "test.oga"
    mock_file.download_to_drive = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 123456

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)
    mock_context.bot.send_chat_action = AsyncMock()

    # Read the sample voice file content
    with open(sample_ogg, "rb") as f:
        sample_voice_content = f.read()

    # Configure download_to_drive to actually write the file
    mock_file.download_to_drive = AsyncMock(
        side_effect=lambda path: open(path, "wb").write(sample_voice_content)
    )

    # Patch AudioSegment to avoid actual audio processing
    with patch("clients.telegram_bot.AudioSegment") as mock_audio_segment, patch(
        "os.remove"
    ) as mock_remove, patch("os.path.exists", return_value=True):
        # Configure AudioSegment mock
        mock_audio = MagicMock()
        mock_audio.export = MagicMock()
        mock_audio_segment.from_ogg.return_value = mock_audio

        await telegram_bot._voice_handler(mock_update, mock_context)

        mock_file.download_to_drive.assert_called_once()
        mock_audio_segment.from_ogg.assert_called_once()
        mock_audio.export.assert_called_once()
        mock_openai_client.transcribe_audio.assert_called_once()
        mock_openai_client.create_chat_completion.assert_called_once()

        # Verify the user was notified
        assert mock_message.reply_text.call_count >= 3

        # Verify the conversation was updated
        assert 123456 in telegram_bot.conversations
        assert len(telegram_bot.conversations[123456]) >= 2
        assert "This is a test transcription" in [
            msg["content"] for msg in telegram_bot.conversations[123456]
        ]

        # Verify temp files were cleaned up
        assert mock_remove.call_count >= 1
