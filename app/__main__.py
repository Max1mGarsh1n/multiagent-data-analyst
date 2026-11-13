import argparse
import sys

from app.connectors.postgres import CatalogError
from app.rag.retriever import retrieve_schema


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Каталог схемы PostgreSQL для SQL-агента",
        epilog="Настройки через переменные окружения: SOURCE_DATABASE_URL, SOURCE_SCHEMA (bookings), POSTGRES_PORT (5433), POSTGRES_PASSWORD. По умолчанию используется локальная demo. Версия v0 возвращает всю схему независимо от вопроса и top_k.",
    )
    parser.add_argument("question", nargs="?", default="Какие данные доступны?")
    parser.add_argument("--source", default="demo")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--format", choices=("prompt", "json"), default="prompt")
    parser.add_argument("--include-views", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        context = retrieve_schema(args.question, args.source, args.top_k, include_views=args.include_views)
    except (ValueError, CatalogError) as error:
        print(str(error), file=sys.stderr)
        return 1
    output = context.model_dump_json(indent=2) if args.format == "json" else context.to_prompt()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
