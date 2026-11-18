import os

from sqlalchemy import URL, make_url
from sqlalchemy.exc import ArgumentError

from app.connectors.postgres import read_schema
from app.contracts.schema import SchemaContext, TableDoc, qualified_name, quote_identifier
from app.rag.scoring import expand_fk_neighbors, rank_lexical, rrf_merge
from app.rag.vector_store import search_tables_pgvector


DEMO_CATEGORICAL_COLUMNS = {
    ("bookings", "flights", "status"),
    ("bookings", "seats", "fare_conditions"),
    ("bookings", "ticket_flights", "fare_conditions"),
}


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


def source_schema_name() -> str:
    return os.getenv("SOURCE_SCHEMA", "bookings").strip() or "bookings"


def get_catalog(source_id: str, *, include_views: bool = False, include_profiles: bool = True) -> list[TableDoc]:
    """Return the catalog of the configured source."""
    if source_id != "demo":
        raise ValueError("Неизвестный источник. Сейчас доступен source_id='demo'.")
    return read_schema(
        source_url(),
        source_schema_name(),
        include_views=include_views,
        include_profiles=include_profiles,
    )


def _strip_profiles_keep_samples(table: TableDoc, source_id: str) -> TableDoc:
    columns = []
    for column in table.columns:
        profile = column.profile
        is_demo_cat = source_id == "demo" and (table.schema_name, table.name, column.name) in DEMO_CATEGORICAL_COLUMNS
        is_generic_cat = (
            not column.is_pk
            and not column.name.endswith("_id")
            and column.name != "id"
            and ("char" in column.type.lower() or "text" in column.type.lower())
        )
        allowed = is_demo_cat or (source_id != "demo" and is_generic_cat)
        values = (
            [item.value for item in profile.top_values]
            if allowed and profile and not profile.values_truncated and profile.distinct_count is not None and profile.distinct_count <= 50
            else []
        )
        columns.append(column.model_copy(update={"profile": None, "sample_values": values}))
    return table.model_copy(update={"columns": columns, "profile": None})


def _build_join_paths(tables: list[TableDoc]) -> list[str]:
    selected_keys = {(t.schema_name, t.name) for t in tables}
    join_paths = []
    for table in tables:
        source = qualified_name(table.schema_name, table.name)
        for foreign_key in table.foreign_keys:
            if (foreign_key.referred_schema, foreign_key.referred_table) not in selected_keys:
                continue
            target = qualified_name(foreign_key.referred_schema, foreign_key.referred_table)
            conditions = [
                f"{source}.{quote_identifier(column)} = {target}.{quote_identifier(referred_column)}"
                for column, referred_column in zip(foreign_key.columns, foreign_key.referred_columns, strict=True)
            ]
            join_paths.append(" AND ".join(conditions))
    return join_paths


def retrieve_schema(
    question: str,
    source_id: str,
    top_k: int = 8,
    *,
    include_views: bool = False,
    include_profiles: bool = True,
    mode: str | None = None,
) -> SchemaContext:
    """Retrieve the relevant subset of schema tables using hybrid RAG + FK expansion."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Вопрос должен быть непустой строкой.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k должен быть положительным целым числом.")

    catalog = get_catalog(source_id, include_views=include_views, include_profiles=include_profiles)
    prepared = [_strip_profiles_keep_samples(table, source_id) for table in catalog]

    rag_mode = (mode or os.getenv("RAG_MODE", "hybrid")).lower()
    if rag_mode == "full_schema" or len(prepared) <= top_k:
        return SchemaContext(
            source_id=source_id,
            retrieval_mode="full_schema",
            top_k_applied=False,
            tables=prepared,
            join_paths=_build_join_paths(prepared),
        )

    lexical_ranked = [idx for idx, _ in rank_lexical(question, prepared)]
    vector_ranked = None
    if rag_mode in {"hybrid", "pgvector"}:
        vector_ranked = search_tables_pgvector(question, source_id, prepared, top_k=max(top_k, 12))

    if vector_ranked and rag_mode == "pgvector":
        seed_indices = vector_ranked
        effective_mode = "pgvector"
    elif vector_ranked and rag_mode == "hybrid":
        seed_indices = rrf_merge([vector_ranked, lexical_ranked])
        effective_mode = "hybrid"
    else:
        seed_indices = lexical_ranked
        effective_mode = "lexical"

    # Keep ~70% direct hits and allow FK-neighbor expansion to fill remaining top_k slots for JOINs
    seed_count = max(1, min(len(seed_indices), max(1, (top_k * 3) // 4)))
    selected_indices = expand_fk_neighbors(seed_indices[:seed_count], prepared, top_k=top_k)
    selected_tables = [prepared[idx] for idx in selected_indices]

    return SchemaContext(
        source_id=source_id,
        retrieval_mode=effective_mode,
        top_k_applied=True,
        tables=selected_tables,
        join_paths=_build_join_paths(selected_tables),
    )
