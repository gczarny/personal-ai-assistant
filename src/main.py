import sys

from loguru import logger

from clients.openai_client import OpenAIClient
from clients.telegram_bot import TelegramBot
from core.settings import get_settings


def configure_logging() -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="INFO",
        backtrace=True,
        diagnose=True
    )

    logger.add(
        "logs/telegram_bot_{time}.log",
        rotation="1 day",
        retention="2 days",
        level="INFO"
    )


if __name__ == "__main__":
    configure_logging()

    logger.info("Starting main script...")

    settings = get_settings()

    openai_client = OpenAIClient(settings.OPENAI_API_KEY.get_secret_value())
    bot = TelegramBot(settings.TELEGRAM_BOT_TOKEN.get_secret_value(), openai_client)

    bot.run_bot()

    logger.info("Main script finished.")