import argparse
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from app.connectors.postgres import CatalogError
from app.rag.retriever import retrieve_schema


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Каталог схемы PostgreSQL и SQL-агент",
        epilog="Настройки через переменные окружения: SOURCE_DATABASE_URL, SOURCE_SCHEMA (bookings), LLM_BASE_URL, LLM_MODEL.",
    )
    parser.add_argument("question", nargs="?", default="Какие данные доступны?")
    parser.add_argument("--source", default="demo")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--format", choices=("prompt", "json"), default="prompt")
    parser.add_argument("--include-views", action="store_true")
    parser.add_argument("--no-profiles", action="store_true", help="Прочитать только метаданные без выборок строк")
    parser.add_argument("--ask", action="store_true", help="Ответить на вопрос через локальную LLM и PostgreSQL")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        if args.ask:
            from app.agents.sql_agent import answer_question

            response = answer_question(args.question, args.source, top_k=args.top_k)
            output = response.model_dump_json(indent=2) if args.format == "json" else (
                f"{response.answer}\n\nSQL:\n{response.sql or '-'}"
            )
            print(output)
            return 0

        context = retrieve_schema(
            args.question,
            args.source,
            args.top_k,
            include_views=args.include_views,
            include_profiles=not args.no_profiles,
        )
    except (ValueError, CatalogError) as error:
        print(str(error), file=sys.stderr)
        return 1
    output = context.model_dump_json(indent=2) if args.format == "json" else context.to_prompt()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
