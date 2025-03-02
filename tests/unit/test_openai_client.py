from unittest.mock import MagicMock, patch

import pytest

from clients.openai_client import OpenAIClient
from core.exceptions import (
    AudioFileNotFoundError,
    AudioFileTooLargeError,
    APIError,
)


class TestOpenAIClient:
    def setup_method(self):
        self.api_key = "test_api_key"
        self.client = OpenAIClient(api_key=self.api_key)
        self.client.client = MagicMock()  # Mock the internal OpenAI client

    def test_init(self):
        assert self.client.config.api_key == self.api_key
        assert self.client.config.model == "gpt-4o"
        assert self.client.config.temperature == 0.6
        assert self.client.config.max_tokens == 150

    def test_init_missing_api_key(self):
        with pytest.raises(ValueError, match="Missing OpenAI API key"):
            OpenAIClient(api_key="")

    def test_transcribe_audio_success(self, sample_mp3):
        # Mock the OpenAI API response
        mock_transcription = MagicMock()
        mock_transcription.text = "This is a test transcription"
        self.client.client.audio.transcriptions.create.return_value = mock_transcription

        result = self.client.transcribe_audio(sample_mp3)

        assert result.success
        assert result.value == "This is a test transcription"
        assert "file_path" in result.metadata
        assert "file_size_mb" in result.metadata
        self.client.client.audio.transcriptions.create.assert_called_once()

    def test_transcribe_audio_file_not_found(self):
        result = self.client.transcribe_audio("non_existent_file.mp3")
        assert not result.success
        assert isinstance(result.error, AudioFileNotFoundError)
        assert "Audio file not found" in result.error_message

    @patch("os.path.exists")
    @patch("pathlib.Path.stat")
    def test_transcribe_audio_file_too_large(self, mock_stat, mock_exists):
        # Mock file existence and size
        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        # Set file size to 30MB (over the 25MB limit)
        mock_stat_result.st_size = 30 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        result = self.client.transcribe_audio("large_file.mp3")
        assert not result.success
        assert isinstance(result.error, AudioFileTooLargeError)
        assert "exceeds" in result.error_message

    def test_transcribe_audio_api_error(self, sample_mp3):
        # Mock file existence
        with patch("os.path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat:

            # Set file size to be within limits
            mock_stat_result = MagicMock()
            mock_stat_result.st_size = 5 * 1024 * 1024
            mock_stat.return_value = mock_stat_result

            # Simulate an API error
            self.client.client.audio.transcriptions.create.side_effect = Exception(
                "API Error"
            )

            result = self.client.transcribe_audio(sample_mp3)
            assert not result.success
            assert isinstance(result.error, APIError)
            assert "Unexpected error during transcription" in result.error_message

    def test_chat_completion_success(self):
        # Mock the OpenAI API response
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a test response"
        mock_choice.finish_reason = "stop"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        self.client.client.chat.completions.create.return_value = mock_completion

        # Create test messages
        messages = [{"role": "user", "content": "Hello"}]

        result = self.client.create_chat_completion(messages)

        assert result.success
        assert result.value == "This is a test response"
        assert result.metadata["finish_reason"] == "stop"
        self.client.client.chat.completions.create.assert_called_once()

    def test_chat_completion_no_choices(self):
        # Mock the OpenAI API response with no choices
        mock_completion = MagicMock()
        mock_completion.choices = []

        self.client.client.chat.completions.create.return_value = mock_completion

        messages = [{"role": "user", "content": "Hello"}]

        result = self.client.create_chat_completion(messages)

        assert not result.success
        assert "No choices returned" in result.error_message

    def test_chat_completion_authentication_error(self):
        # Simulate an authentication error
        self.client.client.chat.completions.create.side_effect = MagicMock(
            side_effect=Exception("Authentication Error")
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = self.client.create_chat_completion(messages)

        assert not result.success
        assert "Unexpected error" in result.error_message
