import json
import os
from typing import Sequence

from pydantic import BaseModel, Field

from app.connectors.sql_guard import UnsafeSQLError
from app.connectors.sql_runner import SQLExecutionError, execute_readonly_sql
from app.contracts.chat import ChatResponse, QueryResult
from app.llm.client import LLMError, LocalLLMClient
from app.rag.retriever import retrieve_schema, source_schema_name, source_url


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
4. Если результат обрезан (truncated=true), обязательно упомяни, что показаны первые строки выборки."""


def _format_history(history: Sequence[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    # Keep only recent turns to protect the 8K context budget on 4GB VRAM
    return [dict(item) for item in history[-6:]]


def answer_question(
    question: str,
    source_id: str,
    *,
    history: Sequence[dict[str, str]] | None = None,
    llm_client: LocalLLMClient | None = None,
    top_k: int | None = None,
    max_retries: int | None = None,
) -> ChatResponse:
    """Run Schema-RAG -> SQL generation -> read-only execution (with self-correction) -> answer synthesis."""
    effective_top_k = top_k if top_k is not None else int(os.getenv("RAG_TOP_K", "8"))
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

    summary_payload = {
        "question": question,
        "sql": result.sql,
        "columns": result.columns,
        "rows_preview": result.rows[:50],
        "row_count": result.row_count,
        "truncated": result.truncated,
        "elapsed_ms": result.elapsed_ms,
    }

    try:
        answer_text = client.chat_text(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(summary_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=450,
        )
    except LLMError:
        answer_text = f"Запрос выполнен, получено строк: {result.row_count}."

    return ChatResponse(
        answer=answer_text or f"Запрос выполнен, получено строк: {result.row_count}.",
        sql=result.sql,
        result=result,
    )
