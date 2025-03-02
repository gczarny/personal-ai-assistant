# src/core/settings.py
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # TELEGRAM
    TELEGRAM_BOT_TOKEN: SecretStr = Field(..., validation_alias="TELEGRAM_BOT_TOKEN")

    # OPENAI
    OPENAI_API_KEY: SecretStr = Field(..., validation_alias="OPENAI_API_KEY")

    # DATABASE
    DATABASE_URL: str = "sqlite:///conversations.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
