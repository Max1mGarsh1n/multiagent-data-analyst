from sqlalchemy import URL, create_engine, make_url, text
from sqlalchemy.exc import SQLAlchemyError

from app.contracts.schema import ColumnDoc, ForeignKeyDoc, TableDoc, UniqueConstraintDoc


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
           col_description(a.attrelid, a.attnum) AS description
    FROM pg_catalog.pg_attribute a
    JOIN pg_catalog.pg_type t ON t.oid = a.atttypid
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


def read_schema(database_url: str | URL, schema_name: str, *, include_views: bool = False) -> list[TableDoc]:
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
                for column in connection.execute(COLUMNS_SQL, parameters).mappings():
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
                return list(tables.values())
    except SQLAlchemyError:
        raise CatalogError("Не удалось прочитать каталог PostgreSQL. Проверь доступность сервера, учётные данные и права.") from None
    finally:
        engine.dispose()
