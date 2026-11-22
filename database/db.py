"""SQLite connection, schema inspection, and read-only query helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import sqlglot
from sqlglot import exp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "campusmind.db"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_ROW_LIMIT = 100


def get_connection(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open a SQLite connection and enable foreign-key enforcement.

    Read-only connections use SQLite's URI ``mode=ro`` and query-only mode so
    writes remain blocked even if a caller bypasses the SQL AST validation.
    """
    path = Path(db_path).expanduser().resolve()
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(f"Database does not exist: {path}")
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if read_only:
        connection.execute("PRAGMA query_only = ON")
    return connection


def initialize_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> Path:
    """Create the database file and apply the idempotent schema."""
    path = Path(db_path).expanduser().resolve()
    schema_file = Path(schema_path).expanduser().resolve()
    if not schema_file.is_file():
        raise FileNotFoundError(f"Schema file does not exist: {schema_file}")

    connection = get_connection(path)
    try:
        connection.executescript(schema_file.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()
    return path


def get_schema_metadata(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, dict[str, Any]]:
    """Return table columns and foreign keys for SQL generation context."""
    connection = get_connection(db_path, read_only=True)
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        metadata: dict[str, dict[str, Any]] = {}
        for table_row in table_rows:
            table_name = str(table_row["name"])
            # The identifier comes from sqlite_master, not user input.
            quoted_name = '"' + table_name.replace('"', '""') + '"'
            columns = connection.execute(
                f"PRAGMA table_info({quoted_name})"
            ).fetchall()
            foreign_keys = connection.execute(
                f"PRAGMA foreign_key_list({quoted_name})"
            ).fetchall()
            metadata[table_name] = {
                "columns": [
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "not_null": bool(row["notnull"]),
                        "primary_key": bool(row["pk"]),
                    }
                    for row in columns
                ],
                "foreign_keys": [
                    {
                        "column": row["from"],
                        "references_table": row["table"],
                        "references_column": row["to"],
                    }
                    for row in foreign_keys
                ],
            }
        return metadata
    finally:
        connection.close()


def get_schema_context(db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """Render inspected tables, columns, and relationships as prompt context."""
    metadata = get_schema_metadata(db_path)
    sections: list[str] = []
    for table, details in metadata.items():
        column_lines = []
        for column in details["columns"]:
            tags = []
            if column["primary_key"]:
                tags.append("PK")
            if column["not_null"]:
                tags.append("NOT NULL")
            suffix = f" [{', '.join(tags)}]" if tags else ""
            column_lines.append(
                f"  - {column['name']} ({column['type'] or 'unspecified'}){suffix}"
            )
        for foreign_key in details["foreign_keys"]:
            column_lines.append(
                "  - FK: {column} -> {table}.{target}".format(
                    column=foreign_key["column"],
                    table=foreign_key["references_table"],
                    target=foreign_key["references_column"],
                )
            )
        sections.append(f"{table}\n" + "\n".join(column_lines))
    return "\n\n".join(sections)


def _validated_select(sql: str, row_limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Accept one SELECT AST only and add a default row cap if absent."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("A non-empty SELECT query is required.")
    if row_limit < 1:
        raise ValueError("row_limit must be a positive integer.")

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"Invalid SQL: {exc}") from exc

    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")
    expression = statements[0]
    if not isinstance(expression, exp.Select):
        raise ValueError("Only read-only SELECT statements are allowed.")

    forbidden_nodes = {"Insert", "Update", "Delete", "Drop", "Alter", "Pragma", "Attach"}
    if any(type(node).__name__ in forbidden_nodes for node in expression.walk()):
        raise ValueError("The SQL contains a forbidden operation.")

    if expression.args.get("limit") is None:
        expression.set(
            "limit",
            exp.Limit(expression=exp.Literal.number(row_limit)),
        )
    return expression.sql(dialect="sqlite")


def execute_readonly_query(
    sql: str,
    params: Sequence[Any] | dict[str, Any] | None = None,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> pd.DataFrame:
    """Execute a validated, read-only SELECT and return a Pandas DataFrame."""
    safe_sql = _validated_select(sql, row_limit=row_limit)
    connection = get_connection(db_path, read_only=True)
    try:
        return pd.read_sql_query(safe_sql, connection, params=params or ())
    finally:
        connection.close()


def _run_verification_queries(db_path: Path) -> None:
    """Print five small read-only queries to verify seed data and joins."""
    queries = [
        (
            "Departments",
            "SELECT dept_id, dept_name, dept_code FROM departments ORDER BY dept_id",
        ),
        (
            "Students per department",
            "SELECT d.dept_name, COUNT(s.student_id) AS student_count "
            "FROM departments AS d JOIN students AS s ON s.dept_id = d.dept_id "
            "GROUP BY d.dept_id, d.dept_name ORDER BY d.dept_id",
        ),
        (
            "Courses per department",
            "SELECT d.dept_code, COUNT(c.course_id) AS course_count "
            "FROM departments AS d JOIN courses AS c ON c.dept_id = d.dept_id "
            "GROUP BY d.dept_id, d.dept_code ORDER BY d.dept_id",
        ),
        (
            "Low-attendance student/course sample",
            "SELECT s.roll_no, s.full_name, c.course_code, a.attendance_pct "
            "FROM attendance AS a JOIN students AS s ON s.student_id = a.student_id "
            "JOIN courses AS c ON c.course_id = a.course_id "
            "WHERE a.attendance_pct < 75 ORDER BY a.attendance_pct, s.roll_no LIMIT 5",
        ),
        (
            "Average final exam percentage by department",
            "SELECT d.dept_name, ROUND(AVG(100.0 * e.marks_obtained / e.max_marks), 1) "
            "AS avg_final_pct FROM exam_results AS e "
            "JOIN students AS s ON s.student_id = e.student_id "
            "JOIN departments AS d ON d.dept_id = s.dept_id "
            "WHERE e.exam_type = 'Final' GROUP BY d.dept_id, d.dept_name "
            "ORDER BY d.dept_id",
        ),
    ]
    for label, query in queries:
        frame = execute_readonly_query(query, db_path=db_path)
        print(f"\n{label}: {len(frame)} row(s)")
        print(frame.head(3).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover - requested standalone verification
    if __package__ in (None, ""):
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from database.sample_data import seed_sample_data
    else:
        from .sample_data import seed_sample_data

    database_path = seed_sample_data(DEFAULT_DB_PATH)
    print(f"Initialized and seeded database: {database_path}")
    _run_verification_queries(database_path)
