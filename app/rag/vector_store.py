import hashlib
import logging
import os
from functools import lru_cache
from typing import Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.contracts.schema import TableDoc
from app.rag.scoring import build_table_search_text

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # intfloat/multilingual-e5-small


def app_database_url() -> str | None:
    url = os.getenv("APP_DATABASE_URL")
    if url:
        return url
    port = os.getenv("POSTGRES_PORT", "5433")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    return f"postgresql+psycopg://postgres:{password}@127.0.0.1:{port}/analyst"


@lru_cache(maxsize=1)
def _get_cpu_embedder(model_name: str):
    """Load lightweight CPU ONNX embedder so GPU VRAM stays free for the SQL LLM."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name, providers=["CPUExecutionProvider"])


def _format_vector(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def _ensure_table(connection) -> None:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    connection.execute(text(f"""
        CREATE TABLE IF NOT EXISTS schema_rag_embeddings (
            source_id TEXT NOT NULL,
            schema_name TEXT NOT NULL,
            table_name TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            doc_text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (source_id, schema_name, table_name)
        )
    """))


def search_tables_pgvector(
    question: str,
    source_id: str,
    tables: Sequence[TableDoc],
    top_k: int,
) -> list[int] | None:
    """Index changed table docs into pgvector (CPU embeddings) and return ranked indices.

    Returns None if fastembed or the pgvector metadata database is unavailable, allowing
    clean fallback to lexical + FK retrieval.
    """
    db_url = app_database_url()
    if not db_url or not tables:
        return None

    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    try:
        embedder = _get_cpu_embedder(model_name)
    except Exception as error:
        logger.debug("CPU embedder unavailable, falling back to lexical RAG: %s", error)
        return None

    engine = create_engine(db_url, connect_args={"connect_timeout": 3}, hide_parameters=True)
    try:
        with engine.begin() as conn:
            _ensure_table(conn)
            existing = {
                (row.schema_name, row.table_name): row.content_hash
                for row in conn.execute(
                    text("SELECT schema_name, table_name, content_hash FROM schema_rag_embeddings WHERE source_id = :sid"),
                    {"sid": source_id},
                )
            }

            to_embed: list[tuple[TableDoc, str, str]] = []
            for table in tables:
                doc_text = build_table_search_text(table)
                content_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
                if existing.get((table.schema_name, table.name)) != content_hash:
                    to_embed.append((table, doc_text, content_hash))

            if to_embed:
                passages = [f"passage: {doc}" for _, doc, _ in to_embed]
                vectors = list(embedder.embed(passages))
                for (table, doc_text, content_hash), vec in zip(to_embed, vectors, strict=True):
                    conn.execute(
                        text("""
                            INSERT INTO schema_rag_embeddings (source_id, schema_name, table_name, content_hash, doc_text, embedding)
                            VALUES (:sid, :schema, :tname, :chash, :doc, CAST(:emb AS vector))
                            ON CONFLICT (source_id, schema_name, table_name)
                            DO UPDATE SET
                                content_hash = EXCLUDED.content_hash,
                                doc_text = EXCLUDED.doc_text,
                                embedding = EXCLUDED.embedding,
                                updated_at = now()
                        """),
                        {
                            "sid": source_id,
                            "schema": table.schema_name,
                            "tname": table.name,
                            "chash": content_hash,
                            "doc": doc_text,
                            "emb": _format_vector(vec.tolist() if hasattr(vec, "tolist") else vec),
                        },
                    )

            query_vec = next(embedder.embed([f"query: {question}"]))
            vec_literal = _format_vector(query_vec.tolist() if hasattr(query_vec, "tolist") else query_vec)
            rows = conn.execute(
                text("""
                    SELECT schema_name, table_name
                    FROM schema_rag_embeddings
                    WHERE source_id = :sid
                    ORDER BY embedding <=> CAST(:qvec AS vector)
                    LIMIT :lim
                """),
                {"sid": source_id, "qvec": vec_literal, "lim": max(top_k, len(tables))},
            ).all()

        key_to_idx = {(t.schema_name, t.name): idx for idx, t in enumerate(tables)}
        ranked = [key_to_idx[(r.schema_name, r.table_name)] for r in rows if (r.schema_name, r.table_name) in key_to_idx]
        return ranked or None
    except SQLAlchemyError as error:
        logger.debug("pgvector index unavailable, falling back to lexical RAG: %s", error)
        return None
    finally:
        engine.dispose()
