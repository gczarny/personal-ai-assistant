class OpenAIModels:
    """OpenAI models used in the application."""
    GPT_4O = "gpt-4o"
    GPT_O3_MINI = "gpt-o3-mini"
    GPT_4_5_PREVIEW = "gpt-4.5-preview"

    DEFAULT_CHAT_MODEL = GPT_4O

    # Image generation models
    DALL_E_3 = "dall-e-3"
    DALL_E_2 = "dall-e-2"

    DEFAULT_IMAGE_MODEL = DALL_E_3

    WHISPER = "whisper-1"

    DEFAULT_TRANSCRIPTION_MODEL = WHISPER


class ImageSizes:
    """Available image sizes for DALL-E models."""
    SIZE_256 = "256x256"
    SIZE_512 = "512x512"
    SIZE_1024 = "1024x1024"
    SIZE_1792_1024 = "1792x1024"  # Landscape
    SIZE_1024_1792 = "1024x1792"  # Portrait

    DEFAULT = SIZE_1024


class ImageQuality:
    """Available quality options for DALL-E models."""
    STANDARD = "standard"
    HD = "hd"
    DEFAULT = STANDARD


DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 150
DEFAULT_MAX_AUDIO_SIZE_MB = 25.0
DEFAULT_MAX_HISTORY_TOKENS = 4000
