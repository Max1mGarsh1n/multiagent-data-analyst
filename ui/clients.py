import json
import math
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from pydantic import TypeAdapter, ValidationError

from app.connectors.postgres import CatalogError
from app.contracts.chat import ChatResponse, QueryResult, SourceInfo
from app.contracts.schema import TableDoc
from app.rag.retriever import get_catalog
from ui.backend import BackendError


class TestBackend:
    """Explicit UI fixtures. Catalog metadata comes from the configured database."""

    def __init__(self, scenario: str = "Таблица"):
        self.scenario = scenario

    def list_sources(self) -> list[SourceInfo]:
        return [SourceInfo(id="demo", title="PostgreSQL · demo")]

    def get_catalog(self, source_id: str) -> list[TableDoc]:
        try:
            return get_catalog(source_id)
        except (CatalogError, ValueError) as error:
            raise BackendError(str(error)) from None

    def chat(self, question: str, source_id: str, session_id: str) -> ChatResponse:
        if source_id != "demo":
            raise BackendError("Неизвестный источник.")
        if self.scenario == "Ошибка":
            raise BackendError("Тестовая ошибка сервиса. Можно повторить отправку или выбрать другой сценарий.")
        rows = [] if self.scenario == "Пустой результат" else [["Scheduled", 120], ["Cancelled", 4]]
        sql = "SELECT status, count(*) AS flights_count\nFROM bookings.flights\nGROUP BY status\nORDER BY flights_count DESC;"
        return ChatResponse(
            answer="Пример ответа для проверки интерфейса. Показаны заранее подготовленные тестовые данные.",
            sql=sql,
            result=QueryResult(sql=sql, columns=["status", "flights_count"], rows=rows, row_count=len(rows)),
        )


class HttpBackend:
    def __init__(self, base_url: str, timeout: float = 180):
        try:
            parsed = urlsplit(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError
            if parsed.query or parsed.fragment or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
            parsed.port
        except ValueError:
            raise BackendError("Проверь адрес HTTP API и положительный таймаут подключения.") from None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(self.base_url + path, data=body, headers={"Accept": "application/json", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise BackendError(f"API вернул HTTP {error.code}. Проверь состояние сервиса и выбранный источник.") from None
        except (URLError, OSError, ValueError, HTTPException):
            raise BackendError("API недоступен или вернул некорректный JSON. Проверь адрес и состояние сервиса.") from None

    def list_sources(self) -> list[SourceInfo]:
        try:
            sources = TypeAdapter(list[SourceInfo]).validate_python(self._request("/sources"))
            if len({source.id for source in sources}) != len(sources):
                raise BackendError("API вернул повторяющиеся идентификаторы источников.")
            return sources
        except ValidationError:
            raise BackendError("Формат списка источников API не соответствует контракту.") from None

    def get_catalog(self, source_id: str) -> list[TableDoc]:
        try:
            return TypeAdapter(list[TableDoc]).validate_python(self._request(f"/sources/{quote(source_id, safe='')}/catalog"))
        except ValidationError:
            raise BackendError("Формат каталога API не соответствует контракту.") from None

    def chat(self, question: str, source_id: str, session_id: str) -> ChatResponse:
        payload = {"question": question, "source_id": source_id, "session_id": session_id}
        try:
            return ChatResponse.model_validate(self._request("/chat", payload))
        except ValidationError:
            raise BackendError("Формат ответа API не соответствует контракту.") from None
