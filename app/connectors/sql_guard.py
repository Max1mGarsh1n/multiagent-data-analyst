import re

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover - fallback when sqlglot is not installed
    sqlglot = None
    exp = None


class UnsafeSQLError(ValueError):
    """Raised when generated SQL fails read-only safety validation."""


FORBIDDEN_FUNCTIONS = frozenset({
    "pg_sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_rotate_logfile",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "dblink",
    "dblink_exec",
    "dblink_connect",
    "lo_import",
    "lo_export",
    "lo_unlink",
    "set_config",
    "query_to_xml",
})

FORBIDDEN_KEYWORD_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|call|do|merge|vacuum|analyze|reindex|cluster|comment|lock|execute|prepare)\b",
    flags=re.IGNORECASE,
)


def _strip_comments_and_strings(sql: str) -> str:
    """Remove SQL comments and single-quoted string literals for fallback keyword scanning."""
    no_block_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line_comments = re.sub(r"--[^\n]*", " ", no_block_comments)
    return re.sub(r"'(?:''|[^'])*'", "''", no_line_comments)


def validate_readonly_sql(sql: str) -> str:
    """Ensure SQL is a single PostgreSQL read-only SELECT/WITH query without dangerous side effects."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQLError("Пустой SQL-запрос.")

    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Пустой SQL-запрос.")

    if sqlglot is not None and exp is not None:
        try:
            statements = sqlglot.parse(cleaned, read="postgres")
        except sqlglot.errors.ParseError as error:
            raise UnsafeSQLError(f"Синтаксическая ошибка SQL: {error}") from error

        if len(statements) != 1 or statements[0] is None:
            raise UnsafeSQLError("Разрешён ровно один SQL-запрос.")

        stmt = statements[0]
        allowed_roots = tuple(
            node
            for node in (
                exp.Select,
                exp.Union,
                getattr(exp, "Intersect", None),
                getattr(exp, "Except", None),
                exp.Subquery,
            )
            if node is not None
        )
        if not isinstance(stmt, allowed_roots):
            raise UnsafeSQLError("Разрешены только SELECT-запросы (включая CTE WITH ... SELECT).")

        forbidden_nodes = tuple(
            node
            for node in (
                getattr(exp, "Insert", None),
                getattr(exp, "Update", None),
                getattr(exp, "Delete", None),
                getattr(exp, "Drop", None),
                getattr(exp, "Alter", None),
                getattr(exp, "Create", None),
                getattr(exp, "Command", None),
                getattr(exp, "Merge", None),
                getattr(exp, "Lock", None),
                getattr(exp, "Into", None),
            )
            if node is not None
        )
        for node_type in forbidden_nodes:
            if stmt.find(node_type) is not None:
                raise UnsafeSQLError("Запрос содержит запрещённую операцию изменения или блокировки данных.")

        for func_node in stmt.find_all(exp.Anonymous, exp.Func):
            func_name = (func_node.name or "").lower()
            if func_name in FORBIDDEN_FUNCTIONS:
                raise UnsafeSQLError(f"Вызов системной функции {func_name}() запрещён.")

        # Strip comments/strings check as defense-in-depth against dialect quirks (e.g., SELECT ... INTO)
        stripped = _strip_comments_and_strings(cleaned)
        if re.search(r"\bselect\b.*\binto\b", stripped, flags=re.IGNORECASE | re.DOTALL):
            # Allow column aliases or type casts? SELECT INTO creates a table if INTO precedes FROM
            if stmt.find(exp.Into) is not None:
                raise UnsafeSQLError("SELECT INTO запрещён.")

        return stmt.sql(dialect="postgres")

    # Fallback validation when sqlglot is unavailable
    stripped = _strip_comments_and_strings(cleaned)
    if ";" in stripped:
        raise UnsafeSQLError("Разрешён ровно один SQL-запрос.")
    if not re.match(r"^\s*(select|with|\()\b", stripped, flags=re.IGNORECASE):
        raise UnsafeSQLError("Разрешены только SELECT-запросы.")
    if FORBIDDEN_KEYWORD_RE.search(stripped):
        raise UnsafeSQLError("Обнаружено запрещённое ключевое слово изменения данных или схемы.")
    for fn in FORBIDDEN_FUNCTIONS:
        if re.search(rf"\b{re.escape(fn)}\s*\(", stripped, flags=re.IGNORECASE):
            raise UnsafeSQLError(f"Вызов системной функции {fn}() запрещён.")
    return cleaned
