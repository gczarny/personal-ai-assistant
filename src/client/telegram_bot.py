from loguru import logger
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

class TelegramBot:
    def __init__(self, token: str):
        self.application = ApplicationBuilder().token(token).build()
        self._register_handlers()


    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler("start", self.start_command))

        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.echo_message)
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

    def run_bot(self) -> None:
        logger.info("Starting the bot with polling...")
        self.application.run_polling()
        logger.info("Bot stopped.")
