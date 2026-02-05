# Personal AI Assistant

Telegram bot serving as a personal AI assistant - combining **ChatGPT**, **DALL-E 3**, **Whisper**, and **web search** in one interface.

## Features

- **Intelligent Conversations** - Chat powered by GPT-4o with conversation history
- **Image Generation** - Create images from text using DALL-E 3
- **Voice Messages** - Send voice notes, get AI responses (Whisper transcription)
- **Image Analysis** - Send images for AI analysis (GPT-4o Vision)
- **Web Search** - Real-time internet search via Tavily API

## Tech Stack

Python 3.12 | python-telegram-bot | OpenAI API | Tavily | SQLAlchemy | Poetry | Pydantic | Loguru

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- [FFmpeg](https://ffmpeg.org/download.html) (for voice messages)
- API Keys: [OpenAI](https://platform.openai.com/api-keys), [Telegram Bot](https://core.telegram.org/bots#how-do-i-create-a-bot), [Tavily](https://tavily.com/) (optional)

## Installation

```bash
git clone https://github.com/gczarny/personal-ai-assistant.git
cd personal-ai-assistant
poetry install
cp .env.example .env
# Edit .env with your API keys
poetry run python src/main.py
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token |
| `TAVILY_API_KEY` | No | Tavily API key for web search |

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Start conversation |
| `/clear` | Clear conversation history |
| `/imagine <prompt>` | Generate image from text |
| `/websearch` | Toggle web search on/off |
| `/search <query>` | Manual web search |

## Tests

```bash
poetry run pytest
```
