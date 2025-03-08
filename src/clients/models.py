from typing import List, Literal

from pydantic import BaseModel, Field

from core.constants import OpenAIModels, ImageSizes, ImageQuality


class OpenAIClientConfig(BaseModel):
    api_key: str
    model: str = OpenAIModels.DEFAULT_CHAT_MODEL
    temperature: float = 0.6
    max_tokens: int = 150
    max_audio_size_mb: float = 25.0
    image_model: str = Field(
        default=OpenAIModels.DEFAULT_IMAGE_MODEL,
        description="Image generation model name",
    )
    image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = (
        ImageSizes.DEFAULT
    )
    image_quality: Literal["standard", "hd"] = ImageQuality.DEFAULT


class VoiceProcessingConfig(BaseModel):
    temp_directory: str = "temp_audio"
    allowed_formats: List[str] = ["mp3", "ogg", "oga", "wav", "m4a"]
    max_duration_seconds: int = 300
