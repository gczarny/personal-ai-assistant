# src/core/utils/token_manager.py
from typing import List, Dict

import tiktoken


class TokenManager:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 4000):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate the number of tokens in the messages."""
        token_count = 0
        for message in messages:
            # Add tokens for message format (role, content start)
            token_count += 4

            # Add tokens for the content
            token_count += len(self.encoding.encode(message.get("content", "")))

        # Add tokens for the completion format
        token_count += 2
        return token_count

    def trim_messages_to_fit(
        self, messages: List[Dict[str, str]], preserve_system: bool = True
    ) -> List[Dict[str, str]]:
        """Trim the messages to fit within token limits while preserving recent context."""
        if not messages:
            return []

        # Always keep system messages if requested
        system_messages = []
        if preserve_system:
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            messages = [msg for msg in messages if msg.get("role") != "system"]

        # Start with most recent messages
        messages.reverse()

        trimmed_messages = []
        current_tokens = self.estimate_tokens(system_messages)

        for message in messages:
            needed = self.estimate_tokens([message])  # <--- Using estimate_tokens here
            if current_tokens + needed <= self.max_tokens:
                trimmed_messages.append(message)
                current_tokens += needed
            else:
                break

        # Restore original order (oldest first)
        trimmed_messages.reverse()
        return system_messages + trimmed_messages
