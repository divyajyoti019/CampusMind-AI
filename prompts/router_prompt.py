"""Multi-label, role-aware intent routing with a constrained JSON response."""

from __future__ import annotations


INTENTS = ("rag", "sql", "analytics")

ROUTER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {"type": "string", "enum": list(INTENTS)},
            "minItems": 1,
        },
        "sub_queries": {
            "type": "object",
            "properties": {
                "rag": {"type": "string"},
                "sql": {"type": "string"},
                "analytics": {"type": "string"},
            },
            "required": list(INTENTS),
            "additionalProperties": False,
        },
        "sql_scope": {
            "type": "string",
            "enum": ["none", "aggregate", "self", "individual"],
        },
    },
    "required": ["intents", "sub_queries", "sql_scope"],
    "additionalProperties": False,
}


def build_router_prompt(query: str, *, role: str) -> str:
    """Describe module capabilities and access policy for structured routing."""
    if not query.strip():
        raise ValueError("query must not be empty")

    role_name = "Student" if role.strip().lower() == "student" else "Faculty / Administrator"
    access_rules = (
        "A Student may use RAG and aggregate analytics. SQL may be routed only for "
        "aggregate summaries unless a trusted student_id is supplied by the server; "
        "never route raw individual records about other students."
        if role_name == "Student"
        else "Faculty / Administrator may use all modules, including individual-record SQL."
    )
    return f"""Classify the user's campus request into one or more modules and rewrite it
as precise, self-contained sub-queries. Keep a sub-query empty when its module is not
selected. Do not answer the request.

Modules:
- rag: policy, handbook, fee, exam-rule, attendance-rule, hostel, or library questions.
- sql: requests for database records, lists, filters, joins, counts, or direct comparisons.
- analytics: KPIs, statistical summaries, charts, distributions, correlations, or
  actionable summaries grounded in database aggregates.

Select every module needed for a compound request. When a request asks for both a
record table and a policy explanation, route to SQL and RAG; include analytics when
a concise statistical summary adds value. Write each module's sub-query so it can be
executed independently without losing important filters or requested fields.

Set sql_scope to:
- none when SQL is not selected;
- aggregate when SQL returns only aggregate/statistical results;
- self when SQL asks for the authenticated user's own records;
- individual when SQL returns named or identifiable student records.

Active role: {role_name}
Access policy: {access_rules}

User request:
{query.strip()}
"""
