"""AST-level read-only SQL validation for the CampusMind SQLite database."""

from __future__ import annotations

from collections.abc import Iterable

import sqlglot
from sqlglot import exp


ALLOWED_TABLES = frozenset(
    {"departments", "students", "courses", "attendance", "exam_results"}
)
DEFAULT_SQL_ROW_LIMIT = 100
STUDENT_RECORD_TABLES = frozenset({"students", "attendance", "exam_results"})
STUDENT_PRIVATE_COLUMNS = frozenset({"student_id", "roll_no", "full_name", "email"})


def validate_and_sanitize_sql(
    sql: str,
    *,
    allowed_tables: Iterable[str] = ALLOWED_TABLES,
    row_limit: int = DEFAULT_SQL_ROW_LIMIT,
) -> str:
    """Parse and validate one SQLite SELECT, whitelist tables, and cap rows.

    A missing limit is set to ``row_limit``. Existing limits are capped at the
    same maximum; non-numeric limits and non-main schema qualifiers are rejected.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query must be a non-empty string")
    if row_limit <= 0:
        raise ValueError("row_limit must be positive")

    try:
        statements = [node for node in sqlglot.parse(sql, read="sqlite") if node is not None]
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL could not be parsed as SQLite: {exc}") from exc

    if len(statements) != 1:
        raise ValueError("Only a single SQL statement is allowed")

    query = statements[0]
    if not isinstance(query, exp.Select):
        raise ValueError("Only SELECT statements are allowed")

    # CTE names are query-local relations. Their underlying tables still pass
    # through the same whitelist check below.
    cte_names = {
        cte.alias_or_name.casefold()
        for cte in query.find_all(exp.CTE)
        if cte.alias_or_name
    }
    allowed = {name.casefold() for name in allowed_tables}
    for table in query.find_all(exp.Table):
        table_name = table.name.casefold()
        if table.db and table.db.casefold() != "main":
            raise ValueError("Only tables in the main SQLite database are allowed")
        if table_name not in allowed and table_name not in cte_names:
            raise ValueError(f"Table is not allowed: {table.name}")

    limit_node = query.args.get("limit")
    if limit_node is None:
        query.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))
    else:
        limit_expression = limit_node.args.get("expression")
        if not isinstance(limit_expression, exp.Literal) or not limit_expression.is_number:
            raise ValueError("LIMIT must use an integer literal")
        try:
            requested_limit = int(limit_expression.this)
        except (TypeError, ValueError) as exc:
            raise ValueError("LIMIT must use an integer literal") from exc
        if requested_limit < 0 or requested_limit > row_limit:
            limit_node.set("expression", exp.Literal.number(row_limit))

    return query.sql(dialect="sqlite")


def validate_sql(
    sql: str,
    *,
    allowed_tables: Iterable[str] = ALLOWED_TABLES,
    row_limit: int = DEFAULT_SQL_ROW_LIMIT,
) -> str:
    """Concise alias for :func:`validate_and_sanitize_sql`."""
    return validate_and_sanitize_sql(
        sql, allowed_tables=allowed_tables, row_limit=row_limit
    )


def _direct_tables(select: exp.Select) -> list[exp.Table]:
    """Find table references owned by this SELECT, excluding nested queries."""
    result = []
    for table in select.find_all(exp.Table):
        parent = table.parent
        while parent is not None and not isinstance(parent, exp.Select):
            parent = parent.parent
        if parent is select:
            result.append(table)
    return result


def _is_counted(node: exp.Expression) -> bool:
    """True when a column or star is the direct argument of COUNT/COUNT(DISTINCT)."""
    parent = node.parent
    if isinstance(parent, exp.Distinct):
        parent = parent.parent
    return isinstance(parent, exp.Count)


def enforce_student_query_scope(sql: str, student_id: int | None = None) -> str:
    """Limit student SQL to aggregates or the authenticated student's records.

    When no trusted ``student_id`` is available, sensitive student columns may
    only appear inside aggregate expressions, and may not filter or group rows.
    With an ID, a predicate is added to every SELECT that directly reads a
    student, attendance, or exam-result table, including nested subqueries.
    """
    try:
        query = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL could not be parsed as SQLite: {exc}") from exc
    if not isinstance(query, exp.Select):
        raise ValueError("Student queries must be SELECT statements")

    select_nodes = [node for node in query.walk() if isinstance(node, exp.Select)]
    if student_id is not None:
        if (
            isinstance(student_id, bool)
            or not isinstance(student_id, int)
            or student_id <= 0
        ):
            raise ValueError("A trusted positive student_id is required")
        for select in select_nodes:
            scoped_tables = [
                table
                for table in _direct_tables(select)
                if table.name.casefold() in STUDENT_RECORD_TABLES
            ]
            for table in scoped_tables:
                alias = table.alias_or_name
                predicate = exp.EQ(
                    this=exp.column("student_id", table=alias),
                    expression=exp.Literal.number(student_id),
                )
                where_clause = select.args.get("where")
                if where_clause is None:
                    select.set("where", exp.Where(this=predicate))
                else:
                    where_clause.set(
                        "this",
                        exp.And(this=where_clause.this, expression=predicate),
                    )
        return query.sql(dialect="sqlite")

    for select in select_nodes:
        scoped_tables = [
            table
            for table in _direct_tables(select)
            if table.name.casefold() in STUDENT_RECORD_TABLES
        ]
        if not scoped_tables:
            continue

        has_aggregate = any(
            isinstance(node, exp.AggFunc)
            for projection in select.expressions
            for node in projection.walk()
        )
        if not has_aggregate:
            raise ValueError(
                "Student queries over academic records require an aggregate "
                "or an authenticated student scope"
            )

        for projection in select.expressions:
            if any(True for _ in projection.find_all(exp.GroupConcat)):
                raise ValueError(
                    "Student aggregate queries cannot concatenate individual values"
                )
            for node in projection.walk():
                if isinstance(node, exp.Star) and not _is_counted(node):
                    raise ValueError(
                        "Student aggregate queries cannot select all columns"
                    )
            for column in projection.find_all(exp.Column):
                if column.name.casefold() not in STUDENT_PRIVATE_COLUMNS:
                    continue
                # A personal identifier may only be counted. Any other use,
                # even inside MAX/MIN or a CASE within an aggregate, can reveal
                # or single out one student's record.
                if not _is_counted(column):
                    raise ValueError(
                        "Student aggregate queries cannot return personal identifiers"
                    )

        # Joining facts to students by ID is normal for aggregates; using a
        # personal identifier to filter, group, or order campus-wide results
        # could disclose an individual's record and is therefore rejected.
        for clause_name in ("where", "group", "having", "order"):
            clause = select.args.get(clause_name)
            if clause is not None and any(
                column.name.casefold() in STUDENT_PRIVATE_COLUMNS
                for column in clause.find_all(exp.Column)
            ):
                raise ValueError(
                    "Student aggregate queries cannot filter or group by a "
                    "personal identifier"
                )

    return query.sql(dialect="sqlite")
