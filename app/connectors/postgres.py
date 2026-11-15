from collections import Counter
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from time import monotonic

from sqlalchemy import URL, Text, cast, column as sql_column, create_engine, func, make_url, select, table as sql_table, text
from sqlalchemy.exc import SQLAlchemyError

from app.contracts.schema import ColumnDoc, ColumnProfile, ForeignKeyDoc, TableDoc, TableProfile, UniqueConstraintDoc, ValueCount


TABLES_SQL = text("""
    SELECT c.oid, c.relname, c.relkind, obj_description(c.oid, 'pg_class') AS description
    FROM pg_catalog.pg_class c
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = :schema
      AND (c.relkind IN ('r', 'p') OR (:include_views AND c.relkind IN ('v', 'm')))
      AND has_table_privilege(c.oid, 'SELECT')
    ORDER BY c.relname
""")

COLUMNS_SQL = text("""
    SELECT a.attrelid, a.attname, format_type(a.atttypid, a.atttypmod) AS data_type,
           NOT (a.attnotnull OR t.typnotnull) AS nullable,
           col_description(a.attrelid, a.attnum) AS description,
           COALESCE(bt.typname, t.typname) AS base_type
    FROM pg_catalog.pg_attribute a
    JOIN pg_catalog.pg_type t ON t.oid = a.atttypid
    LEFT JOIN pg_catalog.pg_type bt ON bt.oid = t.typbasetype
    WHERE a.attrelid = ANY(:oids) AND a.attnum > 0 AND NOT a.attisdropped
    ORDER BY a.attrelid, a.attnum
""")

CONSTRAINTS_SQL = text("""
    SELECT c.conrelid, c.conname, c.contype,
           pg_get_constraintdef(c.oid) AS definition,
           ARRAY(SELECT a.attname
                 FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, position)
                 JOIN pg_catalog.pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
                 ORDER BY k.position) AS columns,
           n.nspname AS referred_schema, r.relname AS referred_table,
           ARRAY(SELECT a.attname
                 FROM unnest(c.confkey) WITH ORDINALITY AS k(attnum, position)
                 JOIN pg_catalog.pg_attribute a ON a.attrelid = c.confrelid AND a.attnum = k.attnum
                 ORDER BY k.position) AS referred_columns
    FROM pg_catalog.pg_constraint c
    LEFT JOIN pg_catalog.pg_class r ON r.oid = c.confrelid
    LEFT JOIN pg_catalog.pg_namespace n ON n.oid = r.relnamespace
    WHERE c.conrelid = ANY(:oids) AND c.contype IN ('p', 'f', 'u')
    ORDER BY c.conrelid, c.conname
""")


class CatalogError(RuntimeError):
    """A database catalog could not be read."""


def sample_bounds(values: list[str], base_type: str) -> tuple[str | None, str | None]:
    """Compare supported scalar values using their native ordering."""
    converters = {"date": date.fromisoformat, "timestamp": datetime.fromisoformat, "timestamptz": datetime.fromisoformat, "time": time.fromisoformat}
    converter = Decimal if base_type in {"int2", "int4", "int8", "numeric", "float4", "float8"} else converters.get(base_type)
    if converter is None or not values:
        return None, None
    try:
        pairs = [(converter(value), value) for value in values]
        if converter is Decimal and any(not key.is_finite() for key, _ in pairs):
            return None, None
        return min(pairs, key=lambda item: item[0])[1], max(pairs, key=lambda item: item[0])[1]
    except (ValueError, InvalidOperation, TypeError):
        return None, None


def profile_tables(connection, tables: dict, base_types: dict) -> None:
    """Bound sample size, each statement, and the total profiling time."""
    deadline = monotonic() + 10
    for oid, table in tables.items():
        if table.kind != "table" or not table.columns or monotonic() >= deadline:
            table.profile = TableProfile(status="skipped", message="Представление или исчерпан общий лимит времени.")
            continue
        try:
            with connection.begin_nested():
                timeout_ms = max(1, min(2000, int((deadline - monotonic()) * 1000)))
                connection.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {"timeout": f"{timeout_ms}ms"})
                relation = sql_table(table.name, *(sql_column(item.name) for item in table.columns), schema=table.schema_name)
                query = select(*(func.left(cast(item, Text), 161) for item in relation.c)).select_from(relation).limit(1000)
                rows = connection.execute(query).all()
                profiles = []
                for index, column in enumerate(table.columns):
                    values = [row[index] for row in rows if row[index] is not None]
                    truncated = any(len(value) > 160 for value in values)
                    counts = Counter(value[:160] for value in values)
                    bounds = (None, None) if truncated else sample_bounds(values, base_types[(oid, column.name)])
                    value_limit = 50 if not truncated and len(counts) <= 50 else 5
                    top_values = [ValueCount(value=value, count=count) for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:value_limit]]
                    profiles.append(ColumnProfile(sample_rows=len(rows), null_count=len(rows) - len(values), distinct_count=None if truncated else len(counts), top_values=top_values, values_truncated=truncated, minimum=bounds[0], maximum=bounds[1]))
                for column, profile in zip(table.columns, profiles):
                    column.profile = profile
                table.profile = TableProfile(status="sampled" if rows else "empty", sample_rows=len(rows))
        except SQLAlchemyError:
            table.profile = TableProfile(status="error", message="Статистика недоступна: таймаут, изменение таблицы или недостаточно прав.")


def read_schema(database_url: str | URL, schema_name: str, *, include_views: bool = False, include_profiles: bool = True) -> list[TableDoc]:
    """Read accessible relations in one read-only, repeatable-read transaction."""
    if not schema_name.strip():
        raise ValueError("Имя схемы должно быть непустым.")
    try:
        url = make_url(database_url)
        if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
            raise ValueError("Unsupported database driver")
        engine = create_engine(url.set(drivername="postgresql+psycopg"), connect_args={"connect_timeout": 5}, hide_parameters=True)
    except (SQLAlchemyError, ValueError):
        raise ValueError("Некорректное подключение PostgreSQL.") from None
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL search_path = pg_catalog"))
                connection.execute(text("SET LOCAL DateStyle = 'ISO, YMD'"))
                connection.execute(text("SET LOCAL TIME ZONE 'UTC'"))
                connection.execute(text("SET LOCAL statement_timeout = '10s'"))
                schema = connection.execute(
                    text("SELECT has_schema_privilege(oid, 'USAGE') FROM pg_namespace WHERE nspname = :schema"),
                    {"schema": schema_name},
                ).first()
                if schema is None:
                    raise ValueError(f"Схема {schema_name!r} не найдена.")
                if not schema[0]:
                    raise CatalogError("У пользователя нет доступа к выбранной схеме.")

                relations = connection.execute(TABLES_SQL, {"schema": schema_name, "include_views": include_views}).mappings()
                kinds = {"r": "table", "p": "table", "v": "view", "m": "materialized_view"}
                tables = {}
                for relation in relations:
                    tables[relation["oid"]] = TableDoc(
                        schema_name=schema_name,
                        name=relation["relname"],
                        kind=kinds[relation["relkind"]],
                        description=relation["description"] or "",
                    )
                if not tables:
                    return []
                parameters = {"oids": list(tables)}
                base_types = {}
                for column in connection.execute(COLUMNS_SQL, parameters).mappings():
                    base_types[(column["attrelid"], column["attname"])] = column["base_type"]
                    tables[column["attrelid"]].columns.append(ColumnDoc(
                        name=column["attname"],
                        type=column["data_type"],
                        description=column["description"] or "",
                        nullable=column["nullable"],
                    ))
                for constraint in connection.execute(CONSTRAINTS_SQL, parameters).mappings():
                    table = tables[constraint["conrelid"]]
                    if constraint["contype"] == "p":
                        table.primary_key = constraint["columns"]
                    elif constraint["contype"] == "f":
                        table.foreign_keys.append(ForeignKeyDoc(
                            name=constraint["conname"],
                            columns=constraint["columns"],
                            referred_schema=constraint["referred_schema"],
                            referred_table=constraint["referred_table"],
                            referred_columns=constraint["referred_columns"],
                            definition=constraint["definition"],
                        ))
                    else:
                        table.unique_constraints.append(UniqueConstraintDoc(
                            name=constraint["conname"],
                            columns=constraint["columns"],
                            definition=constraint["definition"],
                        ))
                for table in tables.values():
                    for column in table.columns:
                        column.is_pk = column.name in table.primary_key
                if include_profiles:
                    profile_tables(connection, tables, base_types)
                return list(tables.values())
    except SQLAlchemyError:
        raise CatalogError("Не удалось прочитать каталог PostgreSQL. Проверь доступность сервера, учётные данные и права.") from None
    finally:
        engine.dispose()
