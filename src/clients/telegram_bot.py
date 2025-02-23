import asyncio

from loguru import logger
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from clients.openai_client import OpenAIClient


class TelegramBot:
    def __init__(self, token: str, openai_client: OpenAIClient):
        self.application = ApplicationBuilder().token(token).build()
        self._register_handlers()
        self.openai_client = openai_client


    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler("start", self.start_command))

        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.openai_handler)
        )

    async def echo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: # noqa
        received_text = update.message.text
        logger.info(f"Received message '{received_text}' from chat_id={update.effective_chat.id}")
        await update.message.reply_text(f"Received: {received_text}")

    @staticmethod
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("Handling /start command for chat_id={}", update.effective_chat.id)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Hello! I am your Telegram bot. How can I help you?"
        )

    async def openai_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_msg = update.message.text
        chat_id = update.effective_chat.id

        logger.info(f"Received message '{user_msg}' from chat_id={chat_id}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg}
        ]

        openai_reply = await asyncio.to_thread(self.openai_client.create_chat_completion, messages)

        await update.message.reply_text(openai_reply)

    def run_bot(self) -> None:
        logger.info("Starting the bot with polling...")
        self.application.run_polling()
        logger.info("Bot stopped.")
