import argparse
import sys

from app.connectors.postgres import CatalogError
from app.rag.retriever import retrieve_schema
from app.rag.indexer import IndexError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Каталог схемы PostgreSQL для SQL-агента",
        epilog="Настройки: SOURCE_DATABASE_URL, SOURCE_SCHEMA (bookings), POSTGRES_PORT (5433), POSTGRES_PASSWORD. Сначала выполните индексацию с --index; поиск использует локальную embedding-модель.",
    )
    parser.add_argument("question", nargs="?", default="Какие данные доступны?")
    parser.add_argument("--source", default="demo")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--format", choices=("prompt", "json"), default="prompt")
    parser.add_argument("--include-views", action="store_true")
    parser.add_argument("--no-profiles", action="store_true", help="Прочитать только метаданные без выборок строк")
    parser.add_argument("--index", action="store_true", help="Построить индекс каталога для выбранного источника")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        if args.index:
            from app.rag.indexer import index_catalog, source_fingerprint
            from app.rag.retriever import get_catalog, source_url
            import os
            count = index_catalog(args.source, get_catalog(args.source, include_views=args.include_views, include_profiles=False), source_fingerprint(source_url(), os.getenv("SOURCE_SCHEMA", "bookings")))
            print(f"Проиндексировано объектов: {count}")
            return 0
        context = retrieve_schema(args.question, args.source, args.top_k, include_views=args.include_views, include_profiles=not args.no_profiles)
    except (ValueError, CatalogError, IndexError) as error:
        print(str(error), file=sys.stderr)
        return 1
    output = context.model_dump_json(indent=2) if args.format == "json" else context.to_prompt()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
