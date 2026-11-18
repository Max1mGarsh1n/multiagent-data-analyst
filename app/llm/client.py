import os
from dataclasses import dataclass
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.llm.json_utils import extract_json_object, strip_thinking

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the local LLM endpoint fails or returns invalid output."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            model=os.getenv("LLM_MODEL", "qwen2.5-coder:7b"),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )


class LocalLLMClient:
    """Thin OpenAI-compatible client tuned for local inference servers (Ollama, llama.cpp, LM Studio)."""

    def __init__(self, settings: LLMSettings | None = None, client: OpenAI | None = None):
        self.settings = settings or LLMSettings.from_env()
        self._client = client or OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.timeout,
        )

    def chat_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or self.settings.max_tokens,
            )
        except Exception as error:
            raise LLMError(
                f"Не удалось получить ответ от локальной LLM ({self.settings.model} @ {self.settings.base_url}): {error}"
            ) from error

        content = response.choices[0].message.content if response.choices else ""
        return strip_thinking(content or "")

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[T, str]:
        """Request structured JSON output and validate against a Pydantic schema."""
        content = ""
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or self.settings.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content if response.choices else ""
        except Exception as first_error:
            # Some OpenAI-compatible servers (e.g. older llama-server configs) reject response_format
            try:
                content = self.chat_text(messages, temperature=temperature, max_tokens=max_tokens)
            except LLMError:
                raise LLMError(
                    f"Ошибка обращения к локальной LLM ({self.settings.model}): {first_error}"
                ) from first_error

        raw_json = extract_json_object(content or "")
        try:
            return schema.model_validate_json(raw_json), content or raw_json
        except ValidationError as error:
            raise ValueError(f"JSON от модели не соответствует схеме: {error}") from error
