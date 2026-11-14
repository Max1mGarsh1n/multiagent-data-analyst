from typing import Protocol

from app.contracts.chat import ChatResponse, SourceInfo
from app.contracts.schema import TableDoc


class BackendError(RuntimeError):
    """A backend request could not be completed."""


class BackendClient(Protocol):
    def list_sources(self) -> list[SourceInfo]: ...

    def get_catalog(self, source_id: str) -> list[TableDoc]: ...

    def chat(self, question: str, source_id: str, session_id: str) -> ChatResponse: ...
