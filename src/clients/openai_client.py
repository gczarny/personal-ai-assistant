import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal, Tuple

from openai import OpenAI, AuthenticationError, RateLimitError, Stream

from loguru import logger
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from clients.models import OpenAIClientConfig
from clients.tavily_search import TavilySearchManager
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
        tavily_api_key: Optional[str] = None,
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
        self.tavily_api_key = tavily_api_key
        self.tavily_manager = None
        if tavily_api_key:
            self.tavily_manager = TavilySearchManager(tavily_api_key)
            logger.info("Tavily Search Manager initialized")
        self.available_functions = {"search_web": self.search_web}

    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_web_search: bool = False,
    ) -> ChatCompletionResult:
        try:
            params = self._prepare_completion_params(
                messages, model, temperature, max_tokens, enable_web_search
            )

            effective_model = params["model"]
            effective_temperature = params["temperature"]
            effective_max_tokens = params["max_tokens"]

            completion = self._execute_completion(params)

            if not completion.choices:
                error = NoChoicesError("No choices returned from OpenAI API")
                logger.error(f"No choices error: {error}")
                return ChatCompletionResult.fail(error=error)

            choice = completion.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason
            metadata = {"finish_reason": finish_reason}

            # Check if model wants to call a function
            if (
                hasattr(message, "tool_calls")
                and message.tool_calls
                and enable_web_search
            ):
                logger.info("Model requested to call a function")

                reply, additional_metadata = await self._process_tool_calls(
                    messages,
                    message.tool_calls,
                    effective_model,
                    effective_temperature,
                    effective_max_tokens,
                )

                metadata.update(additional_metadata)

            else:
                reply = message.content.strip() if message.content else ""
                metadata["used_web_search"] = False

            logger.info(
                f"Final reply from OpenAI: {reply[:100]}..."
                if len(reply) > 100
                else f"Final reply from OpenAI: {reply}"
            )

            if finish_reason == "length":
                logger.warning("The completion reached the max_tokens limit.")
                reply += "\n\n [WARNING]: The response was truncated due to max_tokens limit."

            return ChatCompletionResult.ok(reply, metadata=metadata)

        except Exception as e:
            return self._handle_api_error(e)

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

    async def search_web(self, query: str) -> Dict[str, Any]:
        if not self.tavily_manager:
            logger.error(
                "Tavily search manager not initialized, cannot perform web search"
            )
            return {"error": "Web search functionality is not available", "results": []}

        result = await self.tavily_manager.search(query=query)

        if not result.success:
            logger.error(f"Web search failed: {result.error_message}")
            return {"error": result.error_message, "results": []}

        if isinstance(result.value, dict):
            return result.value
        else:
            logger.warning(f"Unexpected result type from search: {type(result.value)}")
            return {
                "query": query,
                "results": [],
                "answer": str(result.value) if result.value is not None else "",
                "error": "Unexpected result format from search API",
            }

    def _prepare_completion_params(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_web_search: bool = False,
    ) -> Dict[str, Any]:
        effective_model = model or self.config.model
        effective_temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        effective_max_tokens = max_tokens or self.config.max_tokens

        params = {
            "model": effective_model,
            "messages": messages,
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
        }

        # Add tools if web search is enabled
        if enable_web_search and self.tavily_manager:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Search the web for current information on a topic or query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to execute. Should be specific and focused on retrieving factual information.",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]
            logger.info("Web search capability enabled for this request")

        return params

    def _execute_completion(
        self, params: Dict[str, Any]
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        logger.info(f"Sending request to OpenAI {params['model']} model.")
        return self.client.chat.completions.create(**params)

    async def _process_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        tool_responses = []
        search_queries = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.info(
                f"Function call requested: {function_name} with args: {function_args}"
            )

            if function_name == "search_web":
                query = function_args.get("query")
                search_queries.append(query)
                search_result = await self.search_web(query)

                tool_responses.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(search_result),
                    }
                )

        if not tool_responses:
            return "", {"used_web_search": False}

        updated_messages = messages.copy()

        original_response = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }

        updated_messages.append(original_response)
        for response in tool_responses:
            updated_messages.append(response)

        logger.info("Making second request to OpenAI with search results")
        params = {
            "model": model,
            "messages": updated_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        second_completion = self._execute_completion(params)

        if not second_completion.choices:
            raise NoChoicesError("No choices returned from second OpenAI API call")

        reply = second_completion.choices[0].message.content.strip()
        metadata = {
            "used_web_search": True,
            "search_queries": search_queries,
            "finish_reason": second_completion.choices[0].finish_reason,
        }

        return reply, metadata

    @staticmethod
    def _handle_api_error(error: Exception) -> ChatCompletionResult:
        error_str = str(error).lower()

        if (
            isinstance(error, AuthenticationError)
            or "authentication" in error_str
            or "auth" in error_str
        ):
            error_obj = APIAuthenticationError("Authentication failed with OpenAI API")
            logger.error(f"Authentication error: {str(error)}")
            return ChatCompletionResult.fail(error=error_obj)

        elif (
            isinstance(error, RateLimitError)
            or "rate limit" in error_str
            or "quota" in error_str
        ):
            error_obj = APIRateLimitError("Rate limit exceeded with OpenAI API")
            logger.error(f"Rate limit error: {str(error)}")
            return ChatCompletionResult.fail(error=error_obj)

        else:
            error_obj = APIError(f"Unexpected error with OpenAI API: {str(error)}")
            logger.error(f"Unexpected error: {str(error)}")
            return ChatCompletionResult.fail(error=error_obj)
