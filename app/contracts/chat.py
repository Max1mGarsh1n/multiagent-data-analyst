from typing import Any

from pydantic import BaseModel, Field, model_validator


class SourceInfo(BaseModel):
    id: str = Field(min_length=1)
    title: str


class QueryResult(BaseModel):
    sql: str = ""
    columns: list[str]
    rows: list[list[Any]]
    dtypes: list[str] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    elapsed_ms: float = Field(default=0, ge=0)
    truncated: bool = False

    @model_validator(mode="after")
    def validate_rows(self) -> "QueryResult":
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Число значений строки должно совпадать с числом колонок.")
        return self


class ChatResponse(BaseModel):
    answer: str
    sql: str | None = None
    result: QueryResult | None = None
