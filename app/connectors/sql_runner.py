import os
from datetime import date, datetime, time
from decimal import Decimal
from time import perf_counter
from typing import Any
from uuid import UUID

from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.connectors.sql_guard import UnsafeSQLError, validate_readonly_sql
from app.contracts.chat import QueryResult


class SQLExecutionError(RuntimeError):
    """Raised when a validated read-only SQL query fails during execution."""


def _serialize_cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [_serialize_cell(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_cell(v) for k, v in value.items()}
    return str(value)


def _infer_dtype(rows: list[list[Any]], col_index: int) -> str:
    for row in rows:
        value = row[col_index]
        if value is not None:
            return type(value).__name__
    return "null"


def execute_readonly_sql(
    sql: str,
    database_url: str | URL,
    *,
    schema_name: str | None = None,
    max_rows: int | None = None,
    statement_timeout_seconds: int = 10,
) -> QueryResult:
    """Validate and run a single read-only SELECT query inside an isolated transaction."""
    safe_sql = validate_readonly_sql(sql)
    limit = max_rows if max_rows is not None else int(os.getenv("SQL_MAX_ROWS", "200"))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_rows должен быть положительным целым числом.")

    wrapped_sql = f"SELECT * FROM (\n{safe_sql}\n) AS _agent_query LIMIT {limit + 1}"
    engine = create_engine(database_url, connect_args={"connect_timeout": 5}, hide_parameters=True)
    started = perf_counter()

    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text(f"SET LOCAL statement_timeout = '{int(statement_timeout_seconds)}s'"))
                connection.execute(text("SET LOCAL DateStyle = 'ISO, YMD'"))
                connection.execute(text("SET LOCAL TIME ZONE 'UTC'"))
                if schema_name and schema_name.strip():
                    connection.execute(
                        text("SELECT set_config('search_path', :sp, true)"),
                        {"sp": f"{schema_name.strip()}, pg_catalog"},
                    )
                else:
                    connection.execute(text("SET LOCAL search_path = pg_catalog"))

                # Execute via driver SQL with no_parameters=True so literals like ':category' or '%' stay intact.
                result = connection.execution_options(no_parameters=True).exec_driver_sql(wrapped_sql)
                columns = list(result.keys())
                fetched = result.fetchmany(limit + 1)

        truncated = len(fetched) > limit
        rows = [[_serialize_cell(cell) for cell in row] for row in fetched[:limit]]
        dtypes = [_infer_dtype(rows, idx) for idx in range(len(columns))]
        elapsed_ms = round((perf_counter() - started) * 1000, 2)

        return QueryResult(
            sql=safe_sql,
            columns=columns,
            rows=rows,
            dtypes=dtypes,
            row_count=len(rows),
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )
    except UnsafeSQLError:
        raise
    except SQLAlchemyError as error:
        cause = error.__cause__ or error
        raise SQLExecutionError(str(cause)) from None
    finally:
        engine.dispose()
