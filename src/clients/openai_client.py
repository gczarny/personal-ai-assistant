from openai import OpenAI

from loguru import logger


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.6, max_tokens: int = 150):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Missing OpenAI API key. "
                             "Please set the OPENAI_API_KEY environment variable or pass api_key.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def create_chat_completion(self, messages: list):
        try:
            logger.info(f"Sending request to OpenAI {self.model} model")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            reply = completion.choices[0].message.content.strip()
            logger.info(f"Reply from openai: {reply}")

            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning("The completion reached the max_tokens limit.")
                reply += "\n\n [WARNING]: The response was truncated due to max_tokens limit."

            return reply
        except Exception as e:
            logger.error(f"Error: {e}")
            return "I'm sorry, but I couldn't process your request."