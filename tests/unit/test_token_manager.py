from utils.token_manager import TokenManager


def test_token_manager_trim_messages():
    """Test that the TokenManager correctly trims messages to fit within token limits."""
    # Create test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there, how can I help you?"},
        {"role": "user", "content": "What's the weather like?"},
        {
            "role": "assistant",
            "content": "I don't have access to current weather data.",
        },
    ]

    token_manager = TokenManager(model_name="gpt-4o", max_tokens=10)

    token_manager.estimate_tokens = lambda msgs: sum(5 for _ in msgs)

    # Trim messages to fit
    trimmed_messages = token_manager.trim_messages_to_fit(messages)

    # Verify that:
    # 1. System message is preserved
    assert trimmed_messages[0]["role"] == "system"

    # 2. Messages are trimmed to fit token limit
    assert len(trimmed_messages) < len(messages)

    # 3. Most recent messages are preserved (after system message)
    assert (
        trimmed_messages[-1]["content"]
        == "I don't have access to current weather data."
    )
