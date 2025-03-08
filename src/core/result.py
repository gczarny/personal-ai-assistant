# src/core/result.py
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Dict, Any


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """
    A result object that represents the outcome of an operation.

    that class implements the Result pattern to handle success and error cases
    without requiring the exception handling for certain expected error conditions.
    """

    success: bool
    value: Optional[T] = None
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, value: T, metadata: Optional[Dict[str, Any]] = None) -> "Result[T]":
        return cls(success=True, value=value, metadata=metadata)

    @classmethod
    def fail(
        cls,
        error: Optional[Exception] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Result[T]":
        return cls(
            success=False,
            error=error,
            error_message=error_message or (str(error) if error else "Unknown error"),
            metadata=metadata,
        )

    def __bool__(self) -> bool:
        return self.success


TranscriptionResult = Result[str]
ChatCompletionResult = Result[str]
ImageGenerationResult = Result[str]
SearchResult = Result[str]
