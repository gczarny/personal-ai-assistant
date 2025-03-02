from typing import List

from pydantic import BaseModel


class OpenAIClientConfig(BaseModel):
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.6
    max_tokens: int = 150
    max_audio_size_mb: float = 25.0


class VoiceProcessingConfig(BaseModel):
    temp_directory: str = "temp_audio"
    allowed_formats: List[str] = ["mp3", "ogg", "oga", "wav", "m4a"]
    max_duration_seconds: int = 300
