import pytest
import base64
from unittest.mock import MagicMock, AsyncMock
from clients.openai_client import OpenAIClient
from clients.telegram_bot import TelegramBot


@pytest.mark.asyncio
async def test_process_image():
    with open("test.jpg", "rb") as f:
        test_image_data = f.read()

    mock_openai_client = MagicMock(spec=OpenAIClient)
    mock_openai_client.create_chat_completion = MagicMock(
        return_value="This is a test response."
    )

    bot = TelegramBot(token="fake_token", openai_client=mock_openai_client)

    caption = "Describe image"
    file_path = "test.jpg"
    response = await bot.process_image(test_image_data, caption, file_path)

    assert response == "This is a test response."
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(test_image_data).decode('utf-8')}"
                    },
                },
            ],
        }
    ]
    mock_openai_client.create_chat_completion.assert_called_once_with(
        expected_messages, model="gpt-4o"
    )
