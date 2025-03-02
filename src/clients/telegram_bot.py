# src/clients/telegram_bot.py
import asyncio
import base64
import os
from io import BytesIO
from typing import Dict, List, Optional

from loguru import logger
from pydub import AudioSegment
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from clients.models import VoiceProcessingConfig
from clients.openai_client import OpenAIClient
from core.exceptions import (
    AudioFileNotFoundError,
    AudioFileTooLargeError,
    APIAuthenticationError,
    APIRateLimitError,
)
from database.connection import Database
from database.repository import ConversationRepository
from utils.token_manager import TokenManager


class TelegramBot:
    """Telegram bot for interacting with OpenAI."""

    def __init__(
        self,
        token: str,
        openai_client: OpenAIClient,
        voice_config: Optional[VoiceProcessingConfig] = None,
        database: Optional[Database] = None,
        max_history_tokens: int = 4000,
    ):
        """
        Initialize the Telegram bot.

        Args:
            token: Telegram bot token
            openai_client: OpenAI client for API interactions
            voice_config: Optional voice processing configuration
        """
        self.application = ApplicationBuilder().token(token).build()
        self.openai_client = openai_client
        self.conversations: Dict[int, List[Dict[str, str]]] = {}
        self.voice_config = voice_config or VoiceProcessingConfig()

        # Database setup
        self.database = database or Database()
        self.database.create_tables()

        # Token management
        self.token_manager = TokenManager(
            model_name=self.openai_client.config.model, max_tokens=max_history_tokens
        )

        # Temporary in-memory cache
        self.conversations: Dict[int, List[Dict[str, str]]] = {}

        os.makedirs(
            os.path.join(os.getcwd(), self.voice_config.temp_directory), exist_ok=True
        )
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register message handlers for different message types."""
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("clear", self._clear_command))

        # TEXT
        self.application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.PHOTO & ~filters.COMMAND, self._text_handler
            )
        )

        # Images
        self.application.add_handler(MessageHandler(filters.PHOTO, self._image_handler))

        # Voice handler
        self.application.add_handler(MessageHandler(filters.VOICE, self._voice_handler))

        # Any other file types
        self.application.add_handler(
            MessageHandler(
                ~filters.TEXT & ~filters.PHOTO & ~filters.VOICE & ~filters.COMMAND,
                self._unsupported_message_handler,
            )
        )

    @staticmethod
    async def _unsupported_message_handler(
        update: Update, context: ContextTypes.DEFAULT_TYPE # noqa
    ) -> None:
        """Handle unsupported message types."""
        await update.message.reply_text(
            "Sorry, I only support text, image, and voice messages."
        )

    async def _clear_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE # noqa
    ) -> None:
        """Handle the /clear command to reset conversation history."""
        chat_id = update.effective_chat.id

        with self.database.session() as session:
            repo = ConversationRepository(session)
            repo.clear_conversation(str(chat_id))

        # Also clear from in-memory storage if it exists
        if chat_id in self.conversations:
            self.conversations[chat_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        await update.message.reply_text(
            "Conversation history has been cleared. What would you like to talk about now?"
        )

    async def _start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else None

        with self.database.session() as session:
            repo = ConversationRepository(session)
            repo.get_or_create_conversation(str(chat_id), user_id)

            # Add system message if it doesn't exist
            messages = repo.get_messages(str(chat_id))
            if not messages or not any(msg.get("role") == "system" for msg in messages):
                repo.add_message(str(chat_id), "system", "You are a helpful assistant.")

        await context.bot.send_message(
            chat_id=chat_id,
            text="Hello! I am your Telegram bot. I can process text, images, and voice messages. How can I help you?",
        )

    async def _text_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle text messages."""
        user_msg = update.message.text
        chat_id = update.effective_chat.id
        user_name = update.effective_user.username if update.effective_user else None

        logger.info(
            f"Received message from chat_id={chat_id}: '{user_msg[:50]}...'"
            if len(user_msg) > 50
            else f"Received message from chat_id={chat_id}: '{user_msg}'"
        )

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        try:
            with self.database.session() as session:
                repo = ConversationRepository(session)
                repo.get_or_create_conversation(str(chat_id), user_name)
                repo.add_message(str(chat_id), "user", user_msg)

                # Get conversation history
                messages = repo.get_messages(str(chat_id))

                # Apply token limit
                trimmed_messages = self.token_manager.trim_messages_to_fit(messages)

            openai_result = await asyncio.to_thread(
                self.openai_client.create_chat_completion, trimmed_messages
            )

            if not openai_result.success:
                error_message = self._get_user_friendly_error_message(
                    openai_result.error
                )
                await update.message.reply_text(error_message)
                return

            with self.database.session() as session:
                repo = ConversationRepository(session)
                repo.add_message(str(chat_id), "assistant", openai_result.value)

            # Update in-memory store for compatibility with voice/image
            if chat_id not in self.conversations:
                self.conversations[chat_id] = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

            # Keep in-memory storage synchronized
            self.conversations[chat_id].append({"role": "user", "content": user_msg})
            self.conversations[chat_id].append(
                {"role": "assistant", "content": openai_result.value}
            )

            await update.message.reply_text(openai_result.value)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(
                "Sorry, something went wrong. Try again later."
            )

    async def _image_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle image messages."""
        chat_id = update.effective_chat.id
        try:
            if not update.message.photo:
                await update.message.reply_text("No image found in the message.")
                return

            photo = update.message.photo[-1]  # Get highest resolution
            caption = update.message.caption or "Describe this image."

            logger.info(
                f"Received image from chat_id={chat_id} with caption: {caption}"
            )

            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )

            file = await context.bot.get_file(photo.file_id)
            file_stream = BytesIO()
            await file.download_to_memory(file_stream)
            file_stream.seek(0)
            image_data = file_stream.read()

            openai_reply = await self._process_image(
                image_data, caption, file.file_path
            )

            await update.message.reply_text(openai_reply)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your image. Please try again later."
            )

    async def _process_image(
        self, image_data: bytes, caption: str, file_path: str
    ) -> str:
        try:
            base64_image = base64.b64encode(image_data).decode("utf-8")

            mime_types = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
            }

            file_extension = file_path.split(".")[-1].lower()
            mime_type = mime_types.get(file_extension)
            if not mime_type:
                raise ValueError(f"Unsupported image format: {file_extension}")

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

            openai_result = await asyncio.to_thread(
                self.openai_client.create_chat_completion, messages, model="gpt-4o"
            )

            if not openai_result.success:
                return self._get_user_friendly_error_message(openai_result.error)

            return openai_result.value

        except Exception as e:
            logger.error(f"Error in process_image: {e}")
            raise

    async def _voice_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle voice messages."""
        chat_id = update.effective_chat.id

        temp_paths = []

        try:
            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )
            await update.message.reply_text("Processing your voice message...")

            voice = update.message.voice

            if (
                voice.duration
                and voice.duration > self.voice_config.max_duration_seconds
            ):
                await update.message.reply_text(
                    f"Voice message is too long (maximum {self.voice_config.max_duration_seconds // 60} minutes). "
                    "Please send a shorter message."
                )
                return

            file = await context.bot.get_file(voice.file_id)

            logger.info(
                f"Received voice message from chat_id={chat_id}, duration={voice.duration}s"
            )

            file_id_safe = voice.file_id.replace(":", "_").replace("/", "_")[:10]

            temp_dir = os.path.join(os.getcwd(), self.voice_config.temp_directory)
            ogg_file_path = os.path.join(
                temp_dir, f"voice_{chat_id}_{file_id_safe}.oga"
            )
            mp3_file_path = os.path.join(
                temp_dir, f"voice_{chat_id}_{file_id_safe}.mp3"
            )

            # Track temporary files for cleanup
            temp_paths = [ogg_file_path, mp3_file_path]

            await file.download_to_drive(ogg_file_path)
            logger.info(f"Downloaded voice message to {ogg_file_path}")

            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )

            audio = AudioSegment.from_ogg(ogg_file_path)
            audio.export(mp3_file_path, format="mp3")
            logger.info(f"Converted voice message to MP3 at {mp3_file_path}")

            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )

            transcription_result = await asyncio.to_thread(
                self.openai_client.transcribe_audio, mp3_file_path
            )

            if not transcription_result.success:
                error_message = self._get_user_friendly_error_message(
                    transcription_result.error,
                    default_message="I couldn't transcribe your voice message. Please try again.",
                )
                await update.message.reply_text(error_message)
                return

            transcribed_text = transcription_result.value
            logger.info(f"Transcribed voice message: '{transcribed_text}'")

            await update.message.reply_text(f'I heard: "{transcribed_text}"')

            if chat_id not in self.conversations:
                self.conversations[chat_id] = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
            self.conversations[chat_id].append(
                {"role": "user", "content": transcribed_text}
            )

            await context.bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )

            completion_result = await asyncio.to_thread(
                self.openai_client.create_chat_completion,
                self.conversations[chat_id],
            )

            if not completion_result.success:
                error_message = self._get_user_friendly_error_message(
                    completion_result.error,
                    default_message="I couldn't process your voice message. Please try again.",
                )
                await update.message.reply_text(error_message)
                return

            self.conversations[chat_id].append(
                {"role": "assistant", "content": completion_result.value}
            )
            await update.message.reply_text(completion_result.value)

        except Exception as e:
            logger.error(f"Error handling voice message: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your voice message. Please try again."
            )
        finally:
            for path in temp_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {path}: {e}")

    @staticmethod
    def _get_user_friendly_error_message(
        error: Optional[Exception] = None,
        default_message: str = "Sorry, something went wrong. Please try again later.",
    ) -> str:
        if error is None:
            return default_message

        if isinstance(error, APIAuthenticationError):
            return (
                "I'm having trouble connecting to my services. Please contact support."
            )

        if isinstance(error, APIRateLimitError):
            return "I'm receiving too many requests right now. Please try again in a few minutes."

        if isinstance(error, AudioFileNotFoundError):
            return (
                "I couldn't find the audio file. Please try sending your message again."
            )

        if isinstance(error, AudioFileTooLargeError):
            return "Your voice message is too large. Please send a shorter message (under 25MB)."

        return default_message

    def run_bot(self) -> None:
        """Start the bot with polling."""
        logger.info("Starting the bot with polling...")
        self.application.run_polling()
        logger.info("Bot stopped.")
