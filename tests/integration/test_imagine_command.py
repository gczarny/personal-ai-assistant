# tests/integration/test_imagine_command.py
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from aiohttp import ClientResponse
from telegram import Update, Message, Chat
from telegram.ext import ContextTypes

from core.result import ImageGenerationResult


@pytest.mark.asyncio
async def test_imagine_command_success(telegram_bot, mock_openai_client):
    """Test the successful execution of the /imagine command."""
    # Configure mock for image generation
    image_result = ImageGenerationResult.ok(
        "https://example.com/generated-image.png",
        metadata={
            "revised_prompt": "A cat playing the piano",
            "model": "dall-e-3",
            "size": "1024x1024",
            "quality": "standard",
        },
    )
    mock_openai_client.generate_image.return_value = image_result

    # Mock the image download response
    mock_response = MagicMock(spec=ClientResponse)
    mock_response.status = 200
    mock_response.__aenter__.return_value = mock_response
    mock_response.read = AsyncMock(return_value=b"fake_image_data")

    # Mock message and update
    mock_message = MagicMock(spec=Message)
    mock_message.text = "/imagine A cat playing the piano"
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345

    # Mock context
    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock()
    mock_context.bot.delete_message = AsyncMock()
    mock_context.bot.send_photo = AsyncMock()

    # Patch aiohttp session to avoid real HTTP requests
    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        await telegram_bot._imagine_command(mock_update, mock_context)

    # Verify the OpenAI client was called correctly
    mock_openai_client.generate_image.assert_called_once_with("A cat playing the piano")

    # Verify the bot sent the image
    mock_context.bot.send_photo.assert_called_once()
    args, kwargs = mock_context.bot.send_photo.call_args
    assert kwargs["chat_id"] == 12345
    assert kwargs["photo"] == b"fake_image_data"
    assert "A cat playing the piano" in kwargs["caption"]

    # Verify status message was deleted
    mock_context.bot.delete_message.assert_called_once()


@pytest.mark.asyncio
async def test_imagine_command_empty_prompt(telegram_bot):
    """Test the /imagine command with an empty prompt."""

    mock_message = MagicMock(spec=Message)
    mock_message.text = "/imagine"  # Empty prompt
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message

    # Mock context
    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await telegram_bot._imagine_command(mock_update, mock_context)

    mock_message.reply_text.assert_called_once()
    call_args = mock_message.reply_text.call_args[0][0]
    assert "provide some description" in call_args.lower()
    assert "Example:" in call_args


@pytest.mark.asyncio
async def test_imagine_command_api_error(telegram_bot, mock_openai_client):
    """Test handling of API errors during image generation."""
    from core.exceptions import APIError

    error = APIError("Test API Error")
    mock_openai_client.generate_image.return_value = ImageGenerationResult.fail(
        error=error
    )

    mock_message = MagicMock(spec=Message)
    mock_message.text = "/imagine A beautiful mountain"
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345

    # Mock context
    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock()

    await telegram_bot._imagine_command(mock_update, mock_context)

    mock_openai_client.generate_image.assert_called_once()

    # The method should call reply_text at least twice:
    # 1. First for the "Generating image..." message
    # 2. Then for the error message
    assert mock_message.reply_text.call_count >= 2

    # Get the last call which should be the error message
    error_message = mock_message.reply_text.call_args_list[-1][0][0]
    assert "couldn't generate" in error_message.lower()


@pytest.mark.asyncio
async def test_imagine_command_image_download_failure(telegram_bot, mock_openai_client):
    """Test handling of image download failures."""
    image_result = ImageGenerationResult.ok(
        "https://example.com/generated-image.png",
        metadata={"revised_prompt": "A cat playing the piano"},
    )
    mock_openai_client.generate_image.return_value = image_result

    mock_response = MagicMock(spec=ClientResponse)
    mock_response.status = 404
    mock_response.__aenter__.return_value = mock_response

    mock_message = MagicMock(spec=Message)
    mock_message.text = "/imagine A cat playing the piano"
    mock_message.reply_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345

    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mock_context.bot = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock()
    mock_context.bot.delete_message = AsyncMock()

    # Patch aiohttp session to simulate download failure
    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        await telegram_bot._imagine_command(mock_update, mock_context)

    # Verify error handling
    mock_message.reply_text.assert_called()
    error_msg = [
        call_args[0][0] for call_args in mock_message.reply_text.call_args_list
    ]
    assert any("trouble downloading" in msg for msg in error_msg)
