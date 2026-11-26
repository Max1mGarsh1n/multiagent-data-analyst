import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import URL, create_engine, text
from sqlalchemy.exc import ProgrammingError

from app.contracts.schema import TableDoc


MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d13f1b27baf31030b7fd040960d60d909913633f"
MODEL_SHA256 = "a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7"
INDEX_REVISION = MODEL_REVISION + ":schema-descriptions-v1"
DIMENSIONS = 768
DESCRIPTIONS = {
    "aircrafts_data": "Самолёты и модели воздушных судов, коды моделей и дальность полёта.",
    "airports_data": "Аэропорты, коды, названия, города, координаты и часовые пояса.",
    "boarding_passes": "Посадочные талоны пассажиров, места и посадка на конкретный рейс.",
    "bookings": "Бронирования, заказы билетов, дата и общая стоимость брони.",
    "flights": "Рейсы и перелёты, номера, даты вылета и прилёта, аэропорты отправления и назначения, статус отменён задержан прибыл.",
    "seats": "Места в самолётах и классы обслуживания эконом бизнес комфорт.",
    "ticket_flights": "Перелёты по билетам, связь билета с рейсом, тариф и стоимость.",
    "tickets": "Билеты пассажиров, номер билета, пассажир и связь с бронированием.",
}


class IndexError(RuntimeError):
    """The semantic index is unavailable or stale."""


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    path = Path(os.environ.get("RAG_MODEL_PATH", "C:/Temp/codex_iis_database/e5-base"))
    try:
        with (path / "model.safetensors").open("rb") as weights:
            digest = hashlib.file_digest(weights, "sha256").hexdigest()
    except OSError:
        raise IndexError("Локальная модель не найдена. Проверь RAG_MODEL_PATH и файлы модели.") from None
    if digest != MODEL_SHA256:
        raise IndexError("Файл модели повреждён или не соответствует выбранной версии.")
    return SentenceTransformer(str(path), device="cpu", local_files_only=True)


def analyst_url() -> URL:
    return URL.create("postgresql+psycopg", username="postgres", password=os.getenv("POSTGRES_PASSWORD", "postgres"), host="127.0.0.1", port=int(os.getenv("POSTGRES_PORT", "5433")), database="analyst")


def source_fingerprint(url: URL, schema: str) -> str:
    identity = json.dumps([url.drivername, url.host, url.port, url.database, url.username, schema], ensure_ascii=False)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def document(table: TableDoc) -> str:
    description = DESCRIPTIONS.get(table.name, "") if table.schema_name == "bookings" else ""
    columns = "; ".join(f"{c.name}: {c.description} ({c.type})" for c in table.columns)
    return f"Таблица {table.schema_name}.{table.name}. {description} {table.description}. Колонки: {columns}"


def vector_literal(values) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def index_catalog(source_id: str, tables: list[TableDoc], fingerprint: str) -> int:
    if not tables:
        raise IndexError("Каталог пуст: индекс не обновлён.")
    tables = [table.model_copy(update={"profile": None, "columns": [column.model_copy(update={"profile": None, "sample_values": []}) for column in table.columns]}) for table in tables]
    texts = [document(table) for table in tables]
    vectors = embedding_model().encode(["passage: " + value for value in texts], normalize_embeddings=True, show_progress_bar=False)
    engine = create_engine(analyst_url(), connect_args={"connect_timeout": 5}, hide_parameters=True)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE IF NOT EXISTS public.schema_search_docs (source_id text NOT NULL, source_fingerprint text NOT NULL, model_revision text NOT NULL, schema_name text NOT NULL, table_name text NOT NULL, document text NOT NULL, table_json jsonb NOT NULL, embedding vector(768) NOT NULL, PRIMARY KEY (source_id, schema_name, table_name))"))
            connection.execute(text("DELETE FROM public.schema_search_docs WHERE source_id = :source"), {"source": source_id})
            for table, value, vector in zip(tables, texts, vectors, strict=True):
                connection.execute(text("INSERT INTO public.schema_search_docs (source_id, source_fingerprint, model_revision, schema_name, table_name, document, table_json, embedding) VALUES (:source, :fingerprint, :revision, :schema, :name, :document, CAST(:data AS jsonb), CAST(:embedding AS vector))"), {"source": source_id, "fingerprint": fingerprint, "revision": INDEX_REVISION, "schema": table.schema_name, "name": table.name, "document": value, "data": table.model_dump_json(), "embedding": vector_literal(vector)})
        return len(tables)
    finally:
        engine.dispose()


def search_catalog(source_id: str, question: str, fingerprint: str, top_k: int, *, include_views: bool = False) -> tuple[list[TableDoc], list[TableDoc]]:
    engine = create_engine(analyst_url(), connect_args={"connect_timeout": 5}, hide_parameters=True)
    try:
        with engine.connect() as connection:
            try:
                rows = connection.execute(text("SELECT source_fingerprint, model_revision, table_json FROM public.schema_search_docs WHERE source_id = :source ORDER BY schema_name, table_name"), {"source": source_id}).mappings().all()
            except ProgrammingError:
                raise IndexError("Индекс не найден или недоступен. Выполни индексацию каталога.") from None
            if not rows or any(row["source_fingerprint"] != fingerprint or row["model_revision"] != INDEX_REVISION for row in rows):
                raise IndexError("Индекс не найден или относится к другому источнику/модели. Выполни индексацию заново.")
            vector = embedding_model().encode(["query: " + question], normalize_embeddings=True, show_progress_bar=False)[0]
            selected = connection.execute(text("SELECT table_json FROM public.schema_search_docs WHERE source_id = :source AND (:include_views OR table_json->>'kind' = 'table') ORDER BY embedding <=> CAST(:embedding AS vector), schema_name, table_name LIMIT :limit"), {"source": source_id, "embedding": vector_literal(vector), "limit": top_k, "include_views": include_views}).scalars().all()
            return [TableDoc.model_validate(value) for value in selected], [TableDoc.model_validate(row["table_json"]) for row in rows]
    finally:
        engine.dispose()
