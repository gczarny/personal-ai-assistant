import asyncio
import base64
from io import BytesIO
from typing import Dict, List

from loguru import logger
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction

from clients.openai_client import OpenAIClient


class TelegramBot:
    def __init__(self, token: str, openai_client: OpenAIClient):
        self.application = ApplicationBuilder().token(token).build()
        self.openai_client = openai_client
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler("start", self.start_command))

        self.application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.PHOTO & ~filters.COMMAND, self.text_handler
            )
        )
        self.application.add_handler(MessageHandler(filters.PHOTO, self.image_handler))

    @staticmethod
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.info("Handling /start command for chat_id={}", update.effective_chat.id)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Hello! I am your Telegram bot. How can I help you?",
        )

    async def text_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_msg = update.message.text
        chat_id = update.effective_chat.id

        logger.info(f"Received message '{user_msg}' from chat_id={chat_id}")

        if chat_id not in self.conversations:
            self.conversations[chat_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        self.conversations[chat_id].append({"role": "user", "content": user_msg})

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        try:
            openai_reply = await asyncio.to_thread(
                self.openai_client.create_chat_completion, self.conversations[chat_id]
            )
            self.conversations[chat_id].append(
                {"role": "assistant", "content": openai_reply}
            )
            await update.message.reply_text(openai_reply)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(
                "Sorry, something went wrong. Try again later."
            )

    async def image_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat_id = update.effective_chat.id
        try:
            photo = update.message.photo[-1]
            caption = update.message.caption or "Describe this image."

            logger.info(
                f"Received image from chat_id={chat_id} with caption: {caption}"
            )

            file = await context.bot.get_file(photo.file_id)
            file_stream = BytesIO()
            await file.download_to_memory(file_stream)
            file_stream.seek(0)
            image_data = file_stream.read()

            base64_image = base64.b64encode(image_data).decode("utf-8")
            file = await context.bot.get_file(photo.file_id)
            file_extension = file.file_path.split(".")[-1].lower()
            if file_extension == "jpg" or file_extension == "jpeg":
                mime_type = "image/jpeg"
            elif file_extension == "png":
                mime_type = "image/png"
            elif file_extension == "gif":
                mime_type = "image/gif"
            else:
                await update.message.reply_text(
                    "Unsupported image format. Please send a JPG, PNG, or GIF file."
                )
                return
            image_content = f"data:{mime_type};base64,{base64_image}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {"type": "image_url", "image_url": {"url": image_content}},
                    ],
                }
            ]

            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )

            openai_reply = await asyncio.to_thread(
                self.openai_client.create_chat_completion, messages, model="gpt-4o"
            )
            await update.message.reply_text(openai_reply)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, something went wrong. Try again later."
            )

    def run_bot(self) -> None:
        logger.info("Starting the bot with polling...")
        self.application.run_polling()
        logger.info("Bot stopped.")
