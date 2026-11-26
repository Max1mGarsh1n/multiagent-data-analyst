import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.agents.sql_agent import answer_question
from app.api import ChatRequest, app, chat
from app.connectors import sql_guard
from app.connectors.sql_guard import UnsafeSQLError, validate_readonly_sql
from app.connectors.sql_runner import execute_readonly_sql
from app.contracts.analysis import AnalysisResult
from app.contracts.chat import QueryResult
from app.contracts.schema import ColumnDoc, SchemaContext, TableDoc
from app.llm.json_utils import extract_json_object, strip_thinking
from app.rag.indexer import IndexError as RagIndexError
from app.rag.retriever import source_schema_name


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
        self.assertEqual(strip_thinking("<thought>abc</thought>Всего 42 рейса."), "Всего 42 рейса.")


class TestSqlGuard(unittest.TestCase):
    def test_sqlglot_is_installed_and_active(self) -> None:
        self.assertIsNotNone(sql_guard.sqlglot)

    def test_allows_select_cte_and_literals_with_colons_and_percent(self) -> None:
        sql = "WITH agg AS (SELECT status, count(*) AS cnt FROM bookings.flights WHERE status LIKE '%Canc%' GROUP BY status) SELECT ':category' AS label, 123::text AS val, * FROM agg;"
        validated = validate_readonly_sql(sql)
        self.assertIn("SELECT", validated.upper())
        self.assertIn("':category'", validated)
        self.assertIn("'%Canc%'", validated)

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
            "SELECT * INTO tmp_tbl FROM bookings.flights",
            "",
        ]:
            with self.assertRaises(UnsafeSQLError, msg=f"Should reject: {bad_sql}"):
                validate_readonly_sql(bad_sql)


class TestSqlRunner(unittest.TestCase):
    def test_executes_driver_sql_without_bind_param_conflict(self) -> None:
        fake_result = MagicMock()
        fake_result.keys.return_value = ["label", "val", "pct"]
        fake_result.fetchmany.return_value = [(":category", "123", "50%"), (":category", "456", "100%")]

        fake_conn = MagicMock()
        fake_conn.exec_driver_sql.return_value = fake_result
        fake_conn.__enter__.return_value = fake_conn
        fake_conn.execution_options.return_value = fake_conn

        fake_engine = MagicMock()
        fake_engine.connect.return_value = fake_conn

        with patch("app.connectors.sql_runner.create_engine", return_value=fake_engine):
            query_result = execute_readonly_sql(
                "SELECT ':category' AS label, 123::text AS val, '50%' AS pct",
                "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/demo",
                schema_name="bookings",
                max_rows=1,
            )

        fake_conn.execution_options.assert_any_call(no_parameters=True)
        fake_conn.exec_driver_sql.assert_called_once()
        executed_sql = fake_conn.exec_driver_sql.call_args.args[0]
        self.assertIn("':category'", executed_sql)
        self.assertTrue(executed_sql.endswith("LIMIT 2"))
        self.assertTrue(query_result.truncated)
        self.assertEqual(query_result.row_count, 1)
        self.assertEqual(query_result.rows, [[":category", "123", "50%"]])


class TestSqlAgentIntegration(unittest.TestCase):
    def _sample_schema_context(self) -> SchemaContext:
        return SchemaContext(
            source_id="demo",
            retrieval_mode="semantic",
            top_k_applied=True,
            tables=[
                TableDoc(
                    schema_name="bookings",
                    name="flights",
                    description="Рейсы",
                    columns=[
                        ColumnDoc(name="flight_id", type="integer", nullable=False, is_pk=True),
                        ColumnDoc(name="status", type="character varying(20)", nullable=False, sample_values=["Scheduled", "Cancelled"]),
                    ],
                    primary_key=["flight_id"],
                )
            ],
            join_paths=[],
        )

    def test_source_schema_name_helper(self) -> None:
        with patch.dict(os.environ, {"SOURCE_SCHEMA": "custom_schema"}):
            self.assertEqual(source_schema_name(), "custom_schema")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(source_schema_name(), "bookings")

    def test_integrates_with_semantic_retriever_and_analyzer(self) -> None:
        schema_ctx = self._sample_schema_context()
        fake_llm = MagicMock()
        fake_plan = MagicMock(can_answer=True, sql='SELECT status, count(*) FROM "bookings"."flights" GROUP BY status', reason="ok")
        fake_llm.chat_json.return_value = (fake_plan, '{"can_answer": true}')

        fake_result = QueryResult(
            sql='SELECT status, COUNT(*) FROM "bookings"."flights" GROUP BY status',
            columns=["status", "count"],
            rows=[["Cancelled", 414], ["Scheduled", 15383]],
            dtypes=["str", "int"],
            row_count=2,
            elapsed_ms=5.0,
            truncated=False,
        )
        expected_analysis = AnalysisResult(
            summary="Отменено 414 рейсов, запланировано 15 383.",
            insights=["Большинство рейсов запланировано."],
            chart_type="bar",
            plotly_json='{"data": [], "layout": {}}',
            warnings=[],
        )
        analyzer = MagicMock(return_value=expected_analysis)

        with (
            patch("app.agents.sql_agent.retrieve_schema", return_value=schema_ctx) as mock_retrieve,
            patch("app.agents.sql_agent.execute_readonly_sql", return_value=fake_result),
        ):
            response = answer_question("Сколько рейсов каждого статуса?", "demo", llm_client=fake_llm, analyzer=analyzer, top_k=3)

        mock_retrieve.assert_called_once_with("Сколько рейсов каждого статуса?", "demo", top_k=3)
        analyzer.assert_called_once_with("Сколько рейсов каждого статуса?", fake_result)
        fake_llm.chat_text.assert_not_called()
        self.assertEqual(response.answer, expected_analysis.summary)
        self.assertEqual(response.sql, fake_result.sql)
        self.assertEqual(response.result, fake_result)
        self.assertEqual(response.analysis, expected_analysis)

    def test_analyzer_failure_keeps_sql_and_result(self) -> None:
        schema_ctx = self._sample_schema_context()
        fake_llm = MagicMock()
        fake_plan = MagicMock(can_answer=True, sql='SELECT count(*) FROM "bookings"."flights"', reason="ok")
        fake_llm.chat_json.return_value = (fake_plan, '{"can_answer": true}')
        fake_result = QueryResult(
            sql='SELECT COUNT(*) FROM "bookings"."flights"',
            columns=["count"],
            rows=[[33121]],
            dtypes=["int"],
            row_count=1,
            elapsed_ms=2.0,
            truncated=False,
        )
        broken_analyzer = MagicMock(side_effect=RuntimeError("plotly failed"))

        with (
            patch("app.agents.sql_agent.retrieve_schema", return_value=schema_ctx),
            patch("app.agents.sql_agent.execute_readonly_sql", return_value=fake_result),
        ):
            response = answer_question("Сколько рейсов?", "demo", llm_client=fake_llm, analyzer=broken_analyzer)

        self.assertIsNone(response.analysis)
        self.assertEqual(response.sql, fake_result.sql)
        self.assertEqual(response.result, fake_result)
        self.assertIn("1", response.answer)

    def test_temporary_summary_passes_preview_warning_when_rows_exceed_50(self) -> None:
        schema_ctx = self._sample_schema_context()
        fake_llm = MagicMock()
        fake_plan = MagicMock(can_answer=True, sql='SELECT flight_id FROM "bookings"."flights"', reason="ok")
        fake_llm.chat_json.return_value = (fake_plan, '{"can_answer": true}')
        fake_llm.chat_text.return_value = "Показаны первые 50 из 150 строк."

        many_rows = [[idx] for idx in range(150)]
        fake_result = QueryResult(
            sql='SELECT flight_id FROM "bookings"."flights"',
            columns=["flight_id"],
            rows=many_rows,
            dtypes=["int"],
            row_count=150,
            elapsed_ms=4.0,
            truncated=False,
        )

        with (
            patch("app.agents.sql_agent.retrieve_schema", return_value=schema_ctx),
            patch("app.agents.sql_agent.execute_readonly_sql", return_value=fake_result),
        ):
            response = answer_question("Покажи рейсы", "demo", llm_client=fake_llm)

        payload_arg = json.loads(fake_llm.chat_text.call_args.args[0][1]["content"])
        self.assertEqual(payload_arg["preview_row_count"], 50)
        self.assertEqual(payload_arg["row_count"], 150)
        self.assertIn("50 из 150", payload_arg["context_warning"])
        self.assertIsNone(response.analysis)

    def test_api_handles_rag_index_error_with_clean_http_error(self) -> None:
        client = TestClient(app)
        with patch("app.api.answer_question", side_effect=RagIndexError("Индекс не найден. Выполни --index.")):
            resp = client.post("/chat", json={"question": "Сколько рейсов?", "source_id": "demo", "session_id": "s1"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("Индекс не найден", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
