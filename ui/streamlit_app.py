import os
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.chat import ChatResponse
from app.contracts.schema import TableDoc, qualified_name
from ui.backend import BackendClient, BackendError
from ui.clients import HttpBackend, TestBackend


def render_response(response: ChatResponse) -> None:
    st.write(response.answer)
    sql = response.sql or (response.result.sql if response.result else None)
    if sql:
        with st.expander("SQL-запрос"):
            st.code(sql, language="sql")
    result = response.result
    if result is not None:
        if result.rows:
            st.dataframe(result.rows, column_config=dict(enumerate(result.columns)), hide_index=True, width="stretch")
            st.caption(f"Получено строк — {result.row_count}")
        else:
            st.info("Запрос выполнен. Строк по выбранным условиям нет.")
        if result.truncated:
            st.warning("Показана часть результата. Ответ сервера ограничен по числу строк.")


def render_chat(client: BackendClient, session: dict) -> None:
    if st.button("Очистить диалог"):
        session["messages"].clear()
        session["id"] = str(uuid4())
    for message in session["messages"]:
        with st.chat_message(message["role"]):
            if "error" in message:
                st.error(message["error"])
            elif message["role"] == "user":
                st.write(message["text"])
            else:
                render_response(message["response"])
    question = st.chat_input("Задай вопрос о данных", max_chars=10000)
    if question and question.strip():
        question = question.strip()
        session["messages"].append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Получаю ответ…"):
                    response = client.chat(question, session["source_id"], session["id"])
                session["messages"].append({"role": "assistant", "response": response})
                render_response(response)
            except BackendError as error:
                session["messages"].append({"role": "assistant", "error": str(error)})
                st.error(str(error))


def render_catalog(client: BackendClient, session: dict) -> None:
    if st.button("Обновить каталог"):
        session.pop("catalog", None)
    if "catalog" not in session:
        try:
            with st.spinner("Читаю схему БД…"):
                session["catalog"] = client.get_catalog(session["source_id"])
        except BackendError as error:
            st.error(str(error))
            return
    tables: list[TableDoc] = session["catalog"]
    if not tables:
        st.info("В выбранной схеме нет доступных объектов.")
        return
    st.caption(f"Объектов — {len(tables)}")
    table = st.selectbox("Таблица или представление", tables, format_func=lambda item: qualified_name(item.schema_name, item.name))
    st.text(table.description)
    rows = []
    for column in table.columns:
        keys = []
        if column.is_pk:
            keys.append("PK")
        if any(column.name in key.columns for key in table.foreign_keys):
            keys.append("FK")
        rows.append({"Колонка": column.name, "Тип": column.type, "Ключ": ", ".join(keys), "NULL": column.nullable, "Описание": column.description})
    st.dataframe(rows, hide_index=True, width="stretch")
    if table.primary_key:
        st.caption("Первичный ключ — " + ", ".join(table.primary_key))
    for key in [*table.foreign_keys, *table.unique_constraints]:
        st.code(key.definition, language="sql")


def main() -> None:
    st.set_page_config(page_title="Анализ данных", page_icon="💬", layout="wide")
    st.title("Анализ данных")
    with st.sidebar:
        mode = st.selectbox("Подключение", ["Тестовый режим", "HTTP API"])
        if mode == "HTTP API":
            endpoint = st.text_input("Адрес API", value=os.getenv("API_BASE_URL", "http://localhost:8000"))
            try:
                client: BackendClient = HttpBackend(endpoint, timeout=float(os.getenv("API_TIMEOUT_SECONDS", "180")))
            except (BackendError, ValueError):
                st.error("Проверь адрес API и API_TIMEOUT_SECONDS.")
                st.stop()
        else:
            endpoint = "test"
            scenario = st.selectbox("Тестовый сценарий чата", ["Таблица", "Пустой результат", "Ошибка"])
            client = TestBackend(scenario)
        st.button("Проверить подключение")
        try:
            sources = client.list_sources()
        except BackendError as error:
            st.error(str(error))
            st.stop()
        if not sources:
            st.info("Сервис пока не вернул источники данных.")
            st.stop()
        source = st.selectbox("Источник", sources, format_func=lambda item: item.title)
    if mode == "Тестовый режим":
        st.info("Тестовый режим. Чат показывает подготовленные примеры ответов и SQL. Каталог читает настроенную PostgreSQL.")
    session_key = (mode, endpoint, source.id)
    sessions = st.session_state.setdefault("sessions", {})
    session = sessions.setdefault(session_key, {"id": str(uuid4()), "source_id": source.id, "messages": []})
    section = st.radio("Раздел", ["Чат", "Каталог схемы"], horizontal=True, label_visibility="collapsed")
    if section == "Чат":
        render_chat(client, session)
    else:
        render_catalog(client, session)


if __name__ == "__main__":
    main()
