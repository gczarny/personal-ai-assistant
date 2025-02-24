from typing import List, Dict

from openai import OpenAI, AuthenticationError, RateLimitError

from loguru import logger


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.6,
        max_tokens: int = 150,
    ):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Missing OpenAI API key.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        try:
            effective_model = model or self.model
            effective_temperature = (
                temperature if temperature is not None else self.temperature
            )
            effective_max_tokens = max_tokens or self.max_tokens

            logger.info(f"Sending request to OpenAI {effective_model} model.")
            completion = self.client.chat.completions.create(
                model=effective_model,
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            if not completion.choices:
                raise ValueError("No choices returned from OpenAI API")

            reply = completion.choices[0].message.content.strip()
            logger.info(f"Reply from openai: {reply}")

            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning("The completion reached the max_tokens limit.")
                reply += "\n\n [WARNING]: The response was truncated due to max_tokens limit."

            return reply
        except AuthenticationError:
            logger.error("Authentication failed. Check your API key.")
            return "Authentication error. Please contact support."
        except RateLimitError:
            logger.error("Rate limit exceeded.")
            return "Rate limit exceeded. Try again later."
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occurred. Please try again."
