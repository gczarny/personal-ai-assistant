import os
import pytest
import asyncio
from unittest.mock import MagicMock
from pydub import AudioSegment

from clients.openai_client import OpenAIClient
from clients.telegram_bot import TelegramBot


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    from core.result import Result

    client = MagicMock(spec=OpenAIClient)
    client.create_chat_completion.return_value = Result.ok("This is a test response")
    client.transcribe_audio.return_value = Result.ok("This is a test transcription")
    return client


@pytest.fixture
def voice_config():
    """Create a voice processing configuration for testing."""
    from clients.models import VoiceProcessingConfig

    return VoiceProcessingConfig(temp_directory="temp_test_audio")


@pytest.fixture
def telegram_bot(mock_openai_client, voice_config):
    """Create a TelegramBot instance with mocked OpenAI client."""
    bot = TelegramBot(
        token="test_token", openai_client=mock_openai_client, voice_config=voice_config
    )

    # Ensure temp directory exists
    os.makedirs(os.path.join(os.getcwd(), voice_config.temp_directory), exist_ok=True)

    yield bot

    temp_dir = os.path.join(os.getcwd(), voice_config.temp_directory)
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def temp_audio_directory():
    """Create and clean up a temporary directory for audio files."""
    test_dir = os.path.join(os.getcwd(), "temp_test_audio")
    os.makedirs(test_dir, exist_ok=True)

    yield test_dir

    # Clean up directory contents after test
    for file in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, file))

    try:
        os.rmdir(test_dir)
    except:
        pass


@pytest.fixture
def sample_mp3(temp_audio_directory):
    """Create a sample MP3 file for testing."""
    mp3_path = os.path.join(temp_audio_directory, "sample.mp3")

    # Create a silent 1-second audio segment
    AudioSegment.silent(duration=1000).export(mp3_path, format="mp3")

    yield mp3_path


@pytest.fixture
def sample_ogg(temp_audio_directory):
    """Create a sample OGG file for testing."""
    ogg_path = os.path.join(temp_audio_directory, "sample.oga")

    # Create a silent 1-second audio segment
    AudioSegment.silent(duration=1000).export(ogg_path, format="ogg")

    yield ogg_path


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
