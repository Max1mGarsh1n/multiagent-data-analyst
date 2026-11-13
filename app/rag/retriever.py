import os

from sqlalchemy import URL, make_url
from sqlalchemy.exc import ArgumentError

from app.connectors.postgres import read_schema
from app.contracts.schema import SchemaContext, TableDoc, qualified_name, quote_identifier


def source_url() -> URL:
    """Resolve the demo connection from process environment variables."""
    try:
        configured_url = os.getenv("SOURCE_DATABASE_URL")
        if configured_url:
            url = make_url(configured_url)
            if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
                raise ValueError("Поддерживается подключение PostgreSQL через psycopg.")
            return url.set(drivername="postgresql+psycopg")
        return URL.create(
            "postgresql+psycopg",
            username="postgres",
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            host="127.0.0.1",
            port=int(os.getenv("POSTGRES_PORT", "5433")),
            database="demo",
        )
    except (ArgumentError, ValueError):
        raise ValueError("Некорректное подключение. Проверь SOURCE_DATABASE_URL или POSTGRES_PORT.") from None


def get_catalog(source_id: str, *, include_views: bool = False) -> list[TableDoc]:
    """Return the catalog of the configured demo source."""
    if source_id != "demo":
        raise ValueError("Неизвестный источник. В v0 доступен source_id='demo'.")
    return read_schema(source_url(), os.getenv("SOURCE_SCHEMA", "bookings"), include_views=include_views)


def retrieve_schema(question: str, source_id: str, top_k: int = 8, *, include_views: bool = False) -> SchemaContext:
    """Return the whole schema in v0; question and top_k do not filter tables yet."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Вопрос должен быть непустой строкой.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k должен быть положительным целым числом.")
    tables = get_catalog(source_id, include_views=include_views)
    join_paths = []
    for table in tables:
        source = qualified_name(table.schema_name, table.name)
        for foreign_key in table.foreign_keys:
            target = qualified_name(foreign_key.referred_schema, foreign_key.referred_table)
            conditions = []
            for column, referred_column in zip(foreign_key.columns, foreign_key.referred_columns, strict=True):
                conditions.append(f"{source}.{quote_identifier(column)} = {target}.{quote_identifier(referred_column)}")
            join_paths.append(" AND ".join(conditions))
    return SchemaContext(source_id=source_id, tables=tables, join_paths=join_paths)
