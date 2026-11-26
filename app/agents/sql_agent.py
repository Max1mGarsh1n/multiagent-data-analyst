import json
import os
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from app.connectors.sql_guard import UnsafeSQLError
from app.connectors.sql_runner import SQLExecutionError, execute_readonly_sql
from app.contracts.analysis import AnalysisResult
from app.contracts.chat import ChatResponse, QueryResult
from app.llm.client import LLMError, LocalLLMClient
from app.rag.retriever import resolve_top_k, retrieve_schema, source_schema_name, source_url

AnalyzerFn = Callable[[str, QueryResult], AnalysisResult]

_PREVIEW_ROW_LIMIT = 50


class SQLPlan(BaseModel):
    can_answer: bool = True
    sql: str = Field(default="")
    reason: str = Field(default="")


SQL_SYSTEM_PROMPT = """Ты — эксперт по PostgreSQL и Text-to-SQL.
Твоя задача: по схеме базы данных и вопросу пользователя сформировать ОДИН корректный read-only SQL-запрос.

Отвечай СТРОГО в JSON формате без лишнего текста:
{
  "can_answer": true,
  "sql": "SELECT ...",
  "reason": "краткое пояснение выбора таблиц"
}

Жёсткие правила:
1. Используй ТОЛЬКО таблицы, колонки, значения примеров и JOIN-пути из переданной схемы.
2. Всегда указывай квалифицированные имена таблиц со схемой (например, "bookings"."flights").
3. Разрешены только SELECT или WITH ... SELECT. Никаких INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
4. Для строковых категориальных значений используй точные значения из примеров схемы.
5. Если таблицы или колонки для ответа в схеме нет, верни {"can_answer": false, "sql": "", "reason": "объяснение на русском"}."""


ANSWER_SYSTEM_PROMPT = """Ты — аналитик данных. Отвечай на русском языке чётко, кратко и по делу.
Правила:
1. Опирайся ТОЛЬКО на вопрос пользователя, выполненный SQL и полученные строки результата.
2. Не выдумывай числа, категории и выводы, которых нет в данных.
3. Если вернулось 0 строк — прямо напиши, что по заданным условиям данных не найдено.
4. Если передано предупреждение context_warning (или preview_row_count < row_count, или truncated=true), явно укажи, сколько строк показано в превью и что итоги ограничены доступной выборкой."""


def _format_history(history: Sequence[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    # Keep only recent turns to protect the 8K context budget on 4GB VRAM
    return [dict(item) for item in history[-6:]]


def _fallback_answer(result: QueryResult) -> str:
    if result.row_count == 0:
        return "Запрос выполнен. Строк по выбранным условиям не найдено."
    suffix = " (результат ограничен лимитом сервера)" if result.truncated else ""
    return f"Запрос выполнен, получено строк: {result.row_count}{suffix}."


def _summarize_temporarily(question: str, result: QueryResult, client: LocalLLMClient) -> str:
    """Temporary LLM summary fallback until the dedicated analytics module is plugged in."""
    preview_rows = result.rows[:_PREVIEW_ROW_LIMIT]
    preview_row_count = len(preview_rows)
    warnings: list[str] = []
    if result.row_count > preview_row_count:
        warnings.append(
            f"В контекст модели переданы только первые {preview_row_count} из {result.row_count} полученных строк; "
            "ограничь вывод только видимой частью."
        )
    if result.truncated:
        warnings.append("Выборка SQL обрезана лимитом строк сервера (truncated=true).")

    summary_payload = {
        "question": question,
        "sql": result.sql,
        "columns": result.columns,
        "rows_preview": preview_rows,
        "preview_row_count": preview_row_count,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "context_warning": " ".join(warnings) if warnings else None,
        "elapsed_ms": result.elapsed_ms,
    }
    try:
        answer_text = client.chat_text(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(summary_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
    except LLMError:
        return _fallback_answer(result)
    return answer_text or _fallback_answer(result)


def answer_question(
    question: str,
    source_id: str,
    *,
    history: Sequence[dict[str, str]] | None = None,
    llm_client: LocalLLMClient | None = None,
    analyzer: AnalyzerFn | None = None,
    top_k: int | None = None,
    max_retries: int | None = None,
) -> ChatResponse:
    """Run Schema-RAG -> SQL generation -> read-only execution (with self-correction) -> analysis."""
    effective_top_k = resolve_top_k(top_k)
    retries = max_retries if max_retries is not None else int(os.getenv("SQL_MAX_RETRIES", "2"))

    schema_ctx = retrieve_schema(question, source_id, top_k=effective_top_k)
    schema_prompt = schema_ctx.to_prompt()

    client = llm_client or LocalLLMClient()
    db_url = source_url()
    schema_name = source_schema_name()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Схема PostgreSQL:\n{schema_prompt}"},
        *_format_history(history),
        {"role": "user", "content": f"Вопрос пользователя: {question}"},
    ]

    last_error = ""
    candidate_sql: str | None = None
    result: QueryResult | None = None

    for _ in range(retries + 1):
        try:
            plan, raw_response = client.chat_json(messages, SQLPlan, temperature=0.0)
        except ValueError as err:
            last_error = str(err)
            messages.append({
                "role": "user",
                "content": f"Предыдущий ответ не удалось разобрать как JSON ({last_error}). Верни строго JSON: {{\"can_answer\": true, \"sql\": \"...\", \"reason\": \"...\"}}.",
            })
            continue
        except LLMError as err:
            return ChatResponse(answer=str(err))

        if not plan.can_answer or not plan.sql.strip():
            return ChatResponse(
                answer=plan.reason or "По доступной схеме базы данных недостаточно сведений для ответа на этот вопрос.",
            )

        candidate_sql = plan.sql.strip()
        try:
            result = execute_readonly_sql(
                candidate_sql,
                db_url,
                schema_name=schema_name,
            )
            break
        except (UnsafeSQLError, SQLExecutionError) as err:
            last_error = str(err)
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    f"SQL не прошёл проверку или завершился ошибкой PostgreSQL:\n{last_error}\n"
                    "Исправь SQL по схеме и верни новый JSON."
                ),
            })

    if result is None:
        return ChatResponse(
            answer=(
                f"Не удалось выполнить корректный SQL-запрос после {retries + 1} попыток.\n"
                f"Последняя ошибка: {last_error}"
            ),
            sql=candidate_sql,
        )

    if analyzer is not None:
        try:
            analysis = analyzer(question, result)
            return ChatResponse(
                answer=analysis.summary,
                sql=result.sql,
                result=result,
                analysis=analysis,
            )
        except Exception:
            return ChatResponse(
                answer=_fallback_answer(result),
                sql=result.sql,
                result=result,
                analysis=None,
            )

    answer_text = _summarize_temporarily(question, result, client)
    return ChatResponse(
        answer=answer_text,
        sql=result.sql,
        result=result,
        analysis=None,
    )
