import os
from pathlib import Path
from typing import List, Dict, Optional, Any

from openai import OpenAI, AuthenticationError, RateLimitError

from loguru import logger

from clients.models import OpenAIClientConfig
from core.exceptions import (
    NoChoicesError,
    APIAuthenticationError,
    APIRateLimitError,
    APIError,
    AudioFileNotFoundError,
    AudioFileTooLargeError,
)
from core.result import TranscriptionResult, ChatCompletionResult


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.6,
        max_tokens: int = 150,
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
    ) -> str:
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
                    model="whisper-1", file=audio_file
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
