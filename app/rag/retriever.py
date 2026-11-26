import os
from collections import deque

from sqlalchemy import URL, make_url
from sqlalchemy.exc import ArgumentError

from app.connectors.postgres import read_schema
from app.contracts.schema import SchemaContext, TableDoc, qualified_name, quote_identifier


DEMO_CATEGORICAL_COLUMNS = {("bookings", "flights", "status"), ("bookings", "seats", "fare_conditions"), ("bookings", "ticket_flights", "fare_conditions")}


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
    """Resolve the target PostgreSQL schema name from environment variables."""
    return os.getenv("SOURCE_SCHEMA", "bookings").strip() or "bookings"


def resolve_top_k(top_k: int | None = None, *, default: int = 8) -> int:
    """Resolve top_k from explicit value, RAG_TOP_K env var, or default."""
    if top_k is not None:
        return top_k
    raw = os.getenv("RAG_TOP_K")
    if raw is not None and raw.strip():
        try:
            return int(raw)
        except ValueError:
            raise ValueError("RAG_TOP_K должен быть целым числом.") from None
    return default


def get_catalog(source_id: str, *, include_views: bool = False, include_profiles: bool = True) -> list[TableDoc]:
    """Return the catalog of the configured demo source."""
    if source_id != "demo":
        raise ValueError("Неизвестный источник. В v0 доступен source_id='demo'.")
    return read_schema(source_url(), source_schema_name(), include_views=include_views, include_profiles=include_profiles)


def retrieve_schema(question: str, source_id: str, top_k: int = 8, *, include_views: bool = False, include_profiles: bool = True) -> SchemaContext:
    """Find top-k tables and add intermediate tables on foreign-key paths."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Вопрос должен быть непустой строкой.")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k должен быть положительным целым числом.")
    if source_id != "demo":
        raise ValueError("Неизвестный источник. В v1 доступен source_id='demo'.")
    from app.rag.indexer import IndexError, search_catalog, source_fingerprint
    selected, catalog = search_catalog(source_id, question, source_fingerprint(source_url(), source_schema_name()), top_k, include_views=include_views)
    if not include_views:
        selected = [table for table in selected if table.kind == "table"]
        catalog = [table for table in catalog if table.kind == "table"]
    current_catalog = get_catalog(source_id, include_views=include_views, include_profiles=include_profiles)
    metadata = [table.model_copy(update={"profile": None, "columns": [column.model_copy(update={"profile": None, "sample_values": []}) for column in table.columns]}) for table in current_catalog]
    if [table.model_dump() for table in catalog] != [table.model_dump() for table in metadata]:
        raise IndexError("Схема источника изменилась: выполни индексацию заново.")
    by_name = {(t.schema_name, t.name): t for t in catalog}
    adjacency = {key: set() for key in by_name}
    for key, table in by_name.items():
        for fk in table.foreign_keys:
            target = (fk.referred_schema, fk.referred_table)
            if target in by_name:
                adjacency[key].add(target)
                adjacency[target].add(key)
    chosen = {(t.schema_name, t.name) for t in selected}
    roots = sorted(chosen)
    if roots:
        connected = {roots[0]}
        for target in roots[1:]:
            queue = deque([(target, [target])])
            seen = {target}
            while queue:
                node, path = queue.popleft()
                if node in connected:
                    chosen.update(path)
                    connected.update(path)
                    break
                for neighbor in sorted(adjacency[node] - seen):
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    tables = []
    for table in current_catalog:
        if (table.schema_name, table.name) not in chosen:
            continue
        columns = []
        for column in table.columns:
            profile = column.profile
            allowed = source_id == "demo" and (table.schema_name, table.name, column.name) in DEMO_CATEGORICAL_COLUMNS
            values = [item.value for item in profile.top_values] if allowed and profile and not profile.values_truncated and profile.distinct_count is not None and profile.distinct_count <= 50 else []
            columns.append(column.model_copy(update={"profile": None, "sample_values": values}))
        tables.append(table.model_copy(update={"columns": columns, "profile": None}))
    join_paths = []
    for table in tables:
        source = qualified_name(table.schema_name, table.name)
        for foreign_key in table.foreign_keys:
            if (foreign_key.referred_schema, foreign_key.referred_table) not in chosen:
                continue
            target = qualified_name(foreign_key.referred_schema, foreign_key.referred_table)
            conditions = []
            for column, referred_column in zip(foreign_key.columns, foreign_key.referred_columns, strict=True):
                conditions.append(f"{source}.{quote_identifier(column)} = {target}.{quote_identifier(referred_column)}")
            join_paths.append(" AND ".join(conditions))
    return SchemaContext(source_id=source_id, retrieval_mode="semantic", top_k_applied=True, tables=tables, join_paths=join_paths)
