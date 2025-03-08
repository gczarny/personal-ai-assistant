# tests/unit/test_image_generation.py
from unittest.mock import MagicMock

from clients.openai_client import OpenAIClient
from core.constants import OpenAIModels, ImageSizes, ImageQuality
from core.exceptions import (
    ImageGenerationError,
    APIError,
)


class TestImageGeneration:
    def setup_method(self):
        self.api_key = "test_api_key"
        self.client = OpenAIClient(api_key=self.api_key)
        self.client.client = MagicMock()  # Mock the internal OpenAI client

    def test_generate_image_success(self):
        """Test successful image generation."""
        mock_data = MagicMock()
        mock_data.url = "https://example.com/test-image.png"
        mock_data.revised_prompt = "A revised test prompt"

        mock_response = MagicMock()
        mock_response.data = [mock_data]

        self.client.client.images.generate.return_value = mock_response

        prompt = "A test prompt for image generation"

        result = self.client.generate_image(prompt)

        assert result.success is True
        assert result.value == "https://example.com/test-image.png"
        assert "revised_prompt" in result.metadata
        assert result.metadata["revised_prompt"] == "A revised test prompt"
        assert result.metadata["model"] == OpenAIModels.DEFAULT_IMAGE_MODEL
        assert result.metadata["size"] == ImageSizes.DEFAULT
        assert result.metadata["quality"] == ImageQuality.DEFAULT

        self.client.client.images.generate.assert_called_once_with(
            model=OpenAIModels.DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            size=ImageSizes.DEFAULT,
            quality=ImageQuality.DEFAULT,
            n=1,
        )

    def test_generate_image_empty_prompt(self):
        """Test handling of empty prompts."""
        # Test with empty prompt
        result = self.client.generate_image("")

        assert result.success is False
        assert isinstance(result.error, ImageGenerationError)
        assert "Empty or invalid prompt" in result.error_message

        # API should not be called
        self.client.client.images.generate.assert_not_called()

    def test_generate_image_no_data_returned(self):
        """Test handling when API returns no image data."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.data = []

        self.client.client.images.generate.return_value = mock_response

        prompt = "A test prompt for image generation"

        result = self.client.generate_image(prompt)

        assert result.success is False
        assert isinstance(result.error, ImageGenerationError)
        assert "No images generated from API" in result.error_message

    def test_generate_image_authentication_error(self):
        """Test handling of authentication errors."""

        def side_effect(*args, **kwargs):
            raise Exception("Authentication failed with OpenAI API")

        self.client.client.images.generate.side_effect = side_effect

        prompt = "A test prompt for image generation"

        result = self.client.generate_image(prompt)

        assert result.success is False
        assert isinstance(result.error, APIError)
        assert "error during image generation" in result.error_message

    def test_generate_image_rate_limit_error(self):
        """Test handling of rate limit errors."""

        def side_effect(*args, **kwargs):
            raise Exception("Rate limit exceeded with OpenAI API")

        self.client.client.images.generate.side_effect = side_effect

        prompt = "A test prompt for image generation"

        result = self.client.generate_image(prompt)

        assert result.success is False

        assert isinstance(result.error, APIError)
        assert "error during image generation" in result.error_message

    def test_generate_image_unexpected_error(self):
        """Test handling of unexpected errors."""
        # Mock unexpected error
        self.client.client.images.generate.side_effect = Exception("Unexpected error")

        prompt = "A test prompt for image generation"

        result = self.client.generate_image(prompt)

        assert result.success is False
        assert isinstance(result.error, APIError)
        assert "Unexpected error during image generation" in result.error_message

    def test_generate_image_with_custom_parameters(self):
        """Test image generation with custom parameters."""
        # Mock the response
        mock_data = MagicMock()
        mock_data.url = "https://example.com/custom-image.png"

        mock_response = MagicMock()
        mock_response.data = [mock_data]

        self.client.client.images.generate.return_value = mock_response

        # Test input with custom parameters
        prompt = "A custom test prompt"
        size = ImageSizes.SIZE_512
        quality = ImageQuality.HD
        model = OpenAIModels.DALL_E_2

        # Call the function with custom parameters
        result = self.client.generate_image(
            prompt=prompt,
            size=size,
            quality=quality,
            model=model,
        )

        assert result.success is True
        assert result.value == "https://example.com/custom-image.png"
        assert result.metadata["model"] == model
        assert result.metadata["size"] == size
        assert result.metadata["quality"] == quality

        self.client.client.images.generate.assert_called_once_with(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
