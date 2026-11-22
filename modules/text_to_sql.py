"""Gemini-powered, AST-validated, read-only natural-language SQL assistant."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp

from config import settings
from database.db import DEFAULT_DB_PATH, execute_readonly_query, get_schema_context
from prompts.sql_prompt import build_sql_prompt
from utils.security import enforce_student_query_scope, validate_and_sanitize_sql


_FENCED_SQL = re.compile(r"^```(?:sql)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)


def _extract_sql(response: Any) -> str:
    text = (getattr(response, "text", None) or "").strip()
    match = _FENCED_SQL.match(text)
    if match:
        text = match.group(1).strip()
    if not text:
        raise ValueError("Gemini returned an empty SQL statement")
    return text


def _referenced_tables(sql: str) -> list[str]:
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return []
    return sorted({table.name for table in parsed.find_all(exp.Table)})


def _empty_result(explanation: str, error: str) -> dict[str, Any]:
    return {
        "sql": "",
        "dataframe": pd.DataFrame(),
        "explanation": explanation,
        "error": error,
    }


class TextToSQLAssistant:
    """Generate SQL with Gemini, validate it, and execute with a safe DB helper."""

    def __init__(
        self,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
        api_key: str | None = settings.gemini_api_key,
        model: str = settings.gemini_model,
        client: Any | None = None,
        schema_context: str | None = None,
        row_limit: int = settings.sql_row_limit,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.api_key = api_key
        self.model = model
        self.client = client
        self._schema_context = schema_context
        self.row_limit = row_limit

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Set it in the environment or .env."
            )
        from google import genai

        self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _schema(self) -> str:
        if self._schema_context is None:
            self._schema_context = get_schema_context(self.db_path)
        return self._schema_context

    def _generate(self, prompt: str) -> str:
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return _extract_sql(response)

    @staticmethod
    def _correction_prompt(base_prompt: str, previous_sql: str, error: Exception) -> str:
        return f"""{base_prompt}

Your previous SQL was rejected or failed during execution:
{previous_sql}

Error:
{type(error).__name__}: {error}

Correct the query using only the supplied schema. Return exactly one SQLite SELECT
statement and no explanation or markdown. This is the single permitted correction.

SQL:"""

    def query(
        self,
        question: str,
        *,
        role: str = "Faculty / Administrator",
        student_id: int | None = None,
    ) -> dict[str, Any]:
        """Return generated SQL, DataFrame, short explanation, and optional error.

        A validation or execution failure triggers exactly one Gemini correction
        attempt. Invalid SQL never reaches the SQLite executor.
        """
        if not question.strip():
            return _empty_result("Enter a question to query academic records.", "Question is empty")

        try:
            base_prompt = build_sql_prompt(
                question, self._schema(), role=role
            )
        except Exception as exc:
            return _empty_result("Could not prepare the database schema context.", str(exc))

        previous_sql = ""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                prompt = (
                    base_prompt
                    if attempt == 0
                    else self._correction_prompt(base_prompt, previous_sql, last_error)
                )
                generated_sql = self._generate(prompt)
                previous_sql = generated_sql
                safe_sql = validate_and_sanitize_sql(
                    generated_sql, row_limit=self.row_limit
                )
                if role.strip().lower() == "student":
                    safe_sql = enforce_student_query_scope(safe_sql, student_id)
                dataframe = execute_readonly_query(
                    safe_sql,
                    db_path=self.db_path,
                    row_limit=self.row_limit,
                )
                tables = ", ".join(_referenced_tables(safe_sql)) or "the database"
                return {
                    "sql": safe_sql,
                    "dataframe": dataframe,
                    "explanation": (
                        f"Read-only query on {tables} returned "
                        f"{len(dataframe)} row(s)."
                    ),
                    "error": None,
                }
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    break

        message = str(last_error) if last_error else "Unable to generate SQL"
        return {
            "sql": previous_sql,
            "dataframe": pd.DataFrame(),
            "explanation": "The query could not be validated or executed safely.",
            "error": message,
        }


def query_database(
    question: str,
    *,
    role: str = "Faculty / Administrator",
    student_id: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    client: Any | None = None,
) -> dict[str, Any]:
    """Convenience function matching the application module interface."""
    assistant = TextToSQLAssistant(db_path=db_path, client=client)
    return assistant.query(question, role=role, student_id=student_id)
