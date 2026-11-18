from dataclasses import dataclass, field
import unittest

from app.connectors.sql_guard import UnsafeSQLError, validate_readonly_sql
from app.llm.json_utils import extract_json_object, strip_thinking
from app.rag.scoring import expand_fk_neighbors, rank_lexical, rrf_merge, tokenize


@dataclass
class FakeColumn:
    name: str
    type: str = "text"
    description: str = ""
    sample_values: list[str] = field(default_factory=list)


@dataclass
class FakeFK:
    referred_schema: str
    referred_table: str


@dataclass
class FakeTable:
    schema_name: str
    name: str
    description: str = ""
    columns: list[FakeColumn] = field(default_factory=list)
    foreign_keys: list[FakeFK] = field(default_factory=list)


class TestJsonUtils(unittest.TestCase):
    def test_extracts_from_fenced_and_thinking_output(self) -> None:
        raw = '<think>Рассуждаю о схеме...</think>\n```json\n{"can_answer": true, "sql": "SELECT 1", "reason": "ok"}\n```'
        extracted = extract_json_object(raw)
        self.assertEqual(extracted, '{"can_answer": true, "sql": "SELECT 1", "reason": "ok"}')

    def test_extracts_embedded_json_object(self) -> None:
        raw = 'Вот результат:\n{"can_answer": false, "sql": "", "reason": "нет таблицы"}\nУдачи!'
        self.assertEqual(extract_json_object(raw), '{"can_answer": false, "sql": "", "reason": "нет таблицы"}')

    def test_strip_thinking_preserves_plain_answer(self) -> None:
        self.assertEqual(strip_thinking("<think>abc</think>Всего 42 рейса."), "Всего 42 рейса.")


class TestSqlGuard(unittest.TestCase):
    def test_allows_select_and_cte(self) -> None:
        sql = "WITH agg AS (SELECT status, count(*) AS cnt FROM bookings.flights GROUP BY status) SELECT * FROM agg;"
        validated = validate_readonly_sql(sql)
        self.assertIn("SELECT", validated.upper())

    def test_allows_delete_word_inside_string_literal(self) -> None:
        sql = "SELECT * FROM bookings.flights WHERE status = 'delete';"
        validated = validate_readonly_sql(sql)
        self.assertIn("'delete'", validated)

    def test_blocks_mutations_and_multi_statements(self) -> None:
        for bad_sql in [
            "DELETE FROM bookings.flights",
            "UPDATE bookings.flights SET status = 'Cancelled'",
            "DROP TABLE bookings.flights",
            "SELECT 1; DROP TABLE bookings.flights",
            "SELECT pg_sleep(30)",
            "",
        ]:
            with self.assertRaises(UnsafeSQLError, msg=f"Should reject: {bad_sql}"):
                validate_readonly_sql(bad_sql)


class TestRagScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = [
            FakeTable(
                schema_name="bookings",
                name="bookings",
                description="Бронирования билетов",
                columns=[FakeColumn("book_ref"), FakeColumn("total_amount", "numeric")],
            ),
            FakeTable(
                schema_name="bookings",
                name="tickets",
                description="Билеты пассажиров",
                columns=[FakeColumn("ticket_no"), FakeColumn("book_ref"), FakeColumn("passenger_name")],
                foreign_keys=[FakeFK("bookings", "bookings")],
            ),
            FakeTable(
                schema_name="bookings",
                name="ticket_flights",
                description="Перелёты по билетам",
                columns=[FakeColumn("ticket_no"), FakeColumn("flight_id"), FakeColumn("fare_conditions", sample_values=["Economy", "Business"])],
                foreign_keys=[FakeFK("bookings", "tickets"), FakeFK("bookings", "flights")],
            ),
            FakeTable(
                schema_name="bookings",
                name="flights",
                description="Рейсы самолётов",
                columns=[FakeColumn("flight_id"), FakeColumn("status", sample_values=["Scheduled", "Cancelled"])],
            ),
            FakeTable(
                schema_name="bookings",
                name="airports_data",
                description="Справочник аэропортов",
                columns=[FakeColumn("airport_code"), FakeColumn("city")],
            ),
        ]

    def test_tokenize_handles_snake_and_camel_case(self) -> None:
        self.assertEqual(tokenize("fareConditions_test Рейсы"), ["fare", "conditions", "test", "рейсы"])

    def test_lexical_ranking_prefers_matching_tables(self) -> None:
        ranked = rank_lexical("Сколько отменённых рейсов Cancelled status?", self.tables)
        top_table_name = self.tables[ranked[0][0]].name
        self.assertEqual(top_table_name, "flights")

    def test_fk_expansion_pulls_bridge_tables(self) -> None:
        # Seed only tickets (idx=1) -> should expand to bookings (0) and ticket_flights (2)
        expanded = expand_fk_neighbors([1], self.tables, top_k=3)
        self.assertEqual(expanded[0], 1)
        self.assertIn(0, expanded)
        self.assertIn(2, expanded)

    def test_rrf_merge_combines_rankings(self) -> None:
        merged = rrf_merge([[3, 2, 0], [2, 3, 1]])
        self.assertEqual(set(merged[:2]), {2, 3})


if __name__ == "__main__":
    unittest.main()
