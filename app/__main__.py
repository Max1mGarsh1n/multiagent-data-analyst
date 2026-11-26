import argparse
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from app.connectors.postgres import CatalogError
from app.rag.indexer import IndexError
from app.rag.retriever import retrieve_schema


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Каталог схемы PostgreSQL и SQL-агент",
        epilog=(
            "Настройки через переменные окружения: SOURCE_DATABASE_URL, SOURCE_SCHEMA (bookings), "
            "POSTGRES_PORT (5433), POSTGRES_PASSWORD, RAG_MODEL_PATH, RAG_TOP_K, LLM_BASE_URL, LLM_MODEL. "
            "Сначала выполните индексацию с --index; поиск использует локальную embedding-модель."
        ),
    )
    parser.add_argument("question", nargs="?", default="Какие данные доступны?")
    parser.add_argument("--source", default="demo")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--format", choices=("prompt", "json"), default="prompt")
    parser.add_argument("--include-views", action="store_true")
    parser.add_argument("--no-profiles", action="store_true", help="Прочитать только метаданные без выборок строк")
    parser.add_argument("--index", action="store_true", help="Построить индекс каталога для выбранного источника")
    parser.add_argument("--ask", action="store_true", help="Ответить на вопрос через LLM и PostgreSQL")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        if args.index:
            from app.rag.indexer import index_catalog, source_fingerprint
            from app.rag.retriever import get_catalog, source_schema_name, source_url

            count = index_catalog(
                args.source,
                get_catalog(args.source, include_views=args.include_views, include_profiles=False),
                source_fingerprint(source_url(), source_schema_name()),
            )
            print(f"Проиндексировано объектов: {count}")
            return 0

        if args.ask:
            from app.agents.sql_agent import answer_question

            response = answer_question(args.question, args.source, top_k=args.top_k)
            output = response.model_dump_json(indent=2) if args.format == "json" else (
                f"{response.answer}\n\nSQL:\n{response.sql or '-'}"
            )
            print(output)
            return 0

        effective_top_k = args.top_k if args.top_k is not None else 8
        context = retrieve_schema(
            args.question,
            args.source,
            effective_top_k,
            include_views=args.include_views,
            include_profiles=not args.no_profiles,
        )
    except (ValueError, CatalogError, IndexError) as error:
        print(str(error), file=sys.stderr)
        return 1
    output = context.model_dump_json(indent=2) if args.format == "json" else context.to_prompt()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
