class BaseAppException(Exception):
    """Base exception class for application-specific exceptions."""

    def __init__(self, message: str = None, *args):
        self.message = message
        super().__init__(message, *args)


class NoChoicesError(BaseAppException):
    """Raised when no choices are returned from the OpenAI API."""

    pass


class AudioTranscriptionError(BaseAppException):
    """Base class for audio transcription errors."""

    pass


class AudioFileNotFoundError(AudioTranscriptionError):
    """Raised when an audio file cannot be found for transcription."""

    pass


class AudioFileTooLargeError(AudioTranscriptionError):
    """Raised when an audio file exceeds the maximum allowed size."""

    pass


class APIAuthenticationError(BaseAppException):
    """Raised when authentication with an external API fails."""

    pass


class APIRateLimitError(BaseAppException):
    """Raised when an external API rate limit is exceeded."""

    pass


class APIError(BaseAppException):
    """Raised when an error occurs with an external API."""

    pass
