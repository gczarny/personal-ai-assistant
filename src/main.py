import signal
import sys

from loguru import logger

from clients.openai_client import OpenAIClient
from clients.telegram_bot import TelegramBot
from core.settings import get_settings
from database.connection import Database


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
        diagnose=True,
    )

    logger.add(
        "logs/telegram_bot_{time}.log",
        rotation="1 day",
        retention="2 days",
        level="INFO",
    )


if __name__ == "__main__":
    configure_logging()

    logger.info("Starting main script...")

    settings = get_settings()

    database = Database(db_url=settings.DATABASE_URL)
    database.create_tables()

    tavily_api_key = (
        settings.TAVILY_API_KEY.get_secret_value()
        if settings.TAVILY_API_KEY.get_secret_value()
        else None
    )

    if tavily_api_key:
        logger.info("Tavily API key provided - web search functionality enabled")
    else:
        logger.warning("No Tavily API key provided - web search functionality disabled")

    openai_client = OpenAIClient(
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        tavily_api_key=tavily_api_key,
    )
    bot = TelegramBot(
        token=settings.TELEGRAM_BOT_TOKEN.get_secret_value(),
        openai_client=openai_client,
        database=database,
        max_history_tokens=3500,
        enable_web_search=bool(tavily_api_key),
    )

    def shutdown(signum, frame):  # noqa
        logger.info("Received shutdown signal. Stopping bot...")
        bot.application.stop_running()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    bot.run_bot()
