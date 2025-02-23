# src/core/settings.py
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def non_empty_field(**kwargs):
    return Field(..., min_length=1, **kwargs)


class Settings(BaseSettings):
    # TELEGRAM
    TELEGRAM_BOT_TOKEN: SecretStr = Field(..., validation_alias="TELEGRAM_BOT_TOKEN")

    # OPENAI
    OPENAI_API_KEY: SecretStr = Field(..., validation_alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
