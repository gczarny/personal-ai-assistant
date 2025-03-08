import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal

from openai import OpenAI, AuthenticationError, RateLimitError

from loguru import logger

from clients.models import OpenAIClientConfig
from core.constants import (
    OpenAIModels,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    ImageSizes,
    ImageQuality,
)
from core.exceptions import (
    NoChoicesError,
    APIAuthenticationError,
    APIRateLimitError,
    APIError,
    AudioFileNotFoundError,
    AudioFileTooLargeError,
    ImageGenerationError,
)
from core.result import TranscriptionResult, ChatCompletionResult, ImageGenerationResult


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = OpenAIModels.DEFAULT_CHAT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_audio_size_mb: float = 25.0,
    ):
        self.config = OpenAIClientConfig(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_audio_size_mb=max_audio_size_mb,
        )
        if not self.config.api_key:
            raise ValueError("Missing OpenAI API key.")

        self.client = OpenAI(api_key=self.config.api_key)

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletionResult:
        try:
            effective_model = model or self.config.model
            effective_temperature = (
                temperature if temperature is not None else self.config.temperature
            )
            effective_max_tokens = max_tokens or self.config.max_tokens

            logger.info(f"Sending request to OpenAI {effective_model} model.")
            completion = self.client.chat.completions.create(
                model=effective_model,
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            if not completion.choices:
                error = NoChoicesError("No choices returned from OpenAI API")
                logger.error(f"No choices error: {error}")
                return ChatCompletionResult.fail(error=error)

            reply = completion.choices[0].message.content.strip()
            logger.info(
                f"Reply from OpenAI: {reply[:100]}..."
                if len(reply) > 100
                else f"Reply from OpenAI: {reply}"
            )

            finish_reason = completion.choices[0].finish_reason
            metadata = {"finish_reason": finish_reason}

            if finish_reason == "length":
                logger.warning("The completion reached the max_tokens limit.")
                reply += "\n\n [WARNING]: The response was truncated due to max_tokens limit."

            return ChatCompletionResult.ok(reply, metadata=metadata)
        except AuthenticationError as e:
            error = APIAuthenticationError("Authentication failed with OpenAI API")
            logger.error(f"Authentication error: {str(e)}")
            return ChatCompletionResult.fail(error=error)
        except RateLimitError as e:
            error = APIRateLimitError("Rate limit exceeded with OpenAI API")
            logger.error(f"Rate limit error: {str(e)}")
            return ChatCompletionResult.fail(error=error)
        except Exception as e:
            error = APIError(f"Unexpected error with OpenAI API: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")
            return ChatCompletionResult.fail(error=error)

    def transcribe_audio(self, audio_file_path: str) -> TranscriptionResult:
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")

            if not os.path.exists(audio_file_path):
                error = AudioFileNotFoundError(
                    f"Audio file not found: {audio_file_path}"
                )
                logger.error(str(error))
                return TranscriptionResult.fail(error=error)

            file_size_mb = Path(audio_file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_audio_size_mb:
                error = AudioFileTooLargeError(
                    f"Audio file size ({file_size_mb:.2f}MB) exceeds {self.config.max_audio_size_mb}MB limit"
                )
                logger.warning(str(error))
                return TranscriptionResult.fail(error=error)

            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=OpenAIModels.DEFAULT_TRANSCRIPTION_MODEL, file=audio_file
                )

            transcribed_text = transcription.text
            logger.info(f"Transcription result: {transcribed_text}")

            metadata = {"file_size_mb": file_size_mb, "file_path": audio_file_path}

            return TranscriptionResult.ok(transcribed_text, metadata=metadata)

        except AuthenticationError as e:
            error = APIAuthenticationError("Authentication failed during transcription")
            logger.error(f"Authentication error during transcription: {str(e)}")
            return TranscriptionResult.fail(error=error)

        except RateLimitError as e:
            error = APIRateLimitError("Rate limit exceeded during transcription")
            logger.error(f"Rate limit error during transcription: {str(e)}")
            return TranscriptionResult.fail(error=error)

        except Exception as e:
            error = APIError(f"Unexpected error during transcription: {str(e)}")
            logger.error(f"Unexpected error during transcription: {str(e)}")
            return TranscriptionResult.fail(error=error)

    def generate_image(
        self,
        prompt: str,
        size: Literal[
            "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        ] = ImageSizes.DEFAULT,
        quality: Literal["standard", "hd"] = ImageQuality.DEFAULT,
        model: str = OpenAIModels.DEFAULT_IMAGE_MODEL,
        n: int = 1,
    ) -> ImageGenerationResult:
        try:
            logger.info(f"Generating image with prompt: {prompt}")

            if not prompt or prompt.strip() == "":
                error = ImageGenerationError("Empty or invalid prompt provided")
                logger.error(str(error))
                return ImageGenerationResult.fail(error=error)

            response = self.client.images.generate(
                prompt=prompt,
                size=size,
                quality=quality,
                model=model,
                n=n,
            )

            if not response.data or len(response.data) == 0:
                error = ImageGenerationError("No images generated from API")
                logger.error(str(error))
                return ImageGenerationResult.fail(error=error)

            image_url = response.data[0].url

            metadata = {
                "model": model,
                "size": size,
                "quality": quality,
                "revised_prompt": getattr(response.data[0], "revised_prompt", prompt),
            }

            logger.info(f"Successfully generated image, URL: {image_url[:30]}")

            return ImageGenerationResult.ok(image_url, metadata=metadata)

        except AuthenticationError as e:
            error = APIAuthenticationError(
                "Authentication failed with OpenAI image generation API"
            )
            logger.error(f"Authentication error during image generation: {str(e)}")
            return ImageGenerationResult.fail(error=error)

        except RateLimitError as e:
            error = APIRateLimitError(
                "Rate limit exceeded with OpenAI image generation API"
            )
            logger.error(f"Rate limit error during image generation: {str(e)}")
            return ImageGenerationResult.fail(error=error)

        except Exception as e:
            error = APIError(f"Unexpected error during image generation: {str(e)}")
            logger.error(f"Unexpected error during image generation: {str(e)}")
            return ImageGenerationResult.fail(error=error)
