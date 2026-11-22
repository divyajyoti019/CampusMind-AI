"""Gemini structured-output router with deterministic offline fallback and RBAC."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from config import settings
from prompts.router_prompt import INTENTS, ROUTER_RESPONSE_SCHEMA, build_router_prompt


AGGREGATE_CUES = (
    "average",
    "mean",
    "median",
    "count",
    "how many",
    "percentage",
    "total",
    "summary",
    "summarize",
    "statistics",
    "statistical",
    "distribution",
    "correlation",
    "compare",
    "by department",
    "per department",
    "top ",
)
RAW_RECORD_CUES = (
    "list students",
    "show students",
    "which students",
    "student names",
    "student records",
    "roll number",
    "roll no",
    "email",
    "top students",
    "top performers",
    "highest cgpa",
    "best students",
    "students with",
    "student with",
)
SQL_ACTION_CUES = (
    "show",
    "list",
    "find",
    "count",
    "how many",
    "who ",
    "which students",
    "query",
    "records",
    "record",
    "below ",
    "above ",
    "top ",
    "average",
    "mean",
    "median",
    "total",
    "compare",
    "distribution",
    "correlat",
    "percentage",
    "per department",
    "by department",
    "each department",
)
RAG_CUES = (
    "policy",
    "policies",
    "rule",
    "rules",
    "handbook",
    "medical condonation",
    "fee deadline",
    "fee due",
    "hostel",
    "library",
    "according to",
    "university document",
)
SQL_CUES = (
    "students",
    "course",
    "courses",
    "attendance",
    "exam",
    "grade",
    "marks",
    "cgpa",
    "department",
    "record",
    "records",
    "below ",
    "above ",
)
ANALYTICS_CUES = (
    "chart",
    "graph",
    "trend",
    "distribution",
    "analytics",
    "average",
    "mean",
    "compare",
    "summary",
    "summarize",
    "statistics",
    "correlation",
    "correlat",
    "percentage",
    "dashboard",
    "kpi",
    "each department",
)


@dataclass
class RoutePlan:
    """Normalized routing result returned by the router to the Streamlit app."""

    intents: list[str]
    sub_queries: dict[str, str]
    sql_scope: str = "none"
    blocked_intents: list[str] | None = None
    access_message: str | None = None
    route_source: str = "gemini"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntentRouter:
    """Classify and decompose one user request, applying role-based route policy."""

    def __init__(
        self,
        *,
        api_key: str | None = settings.gemini_api_key,
        model: str = settings.gemini_model,
        client: Any | None = None,
        use_llm: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.use_llm = use_llm

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from google import genai

        self.client = genai.Client(api_key=self.api_key)
        return self.client

    @staticmethod
    def _normalize_role(role: str) -> str:
        return "Student" if role.strip().casefold() == "student" else "Faculty / Administrator"

    @staticmethod
    def _is_aggregate_request(query: str) -> bool:
        lowered = query.casefold()
        if IntentRouter._is_raw_student_request(lowered):
            return False
        return any(cue in lowered for cue in AGGREGATE_CUES)

    @staticmethod
    def _is_raw_student_request(lowered_query: str) -> bool:
        if any(cue in lowered_query for cue in RAW_RECORD_CUES):
            return True
        names_action = any(
            cue in lowered_query
            for cue in ("show ", "list ", "who ", "which ", "find ", "top ", "highest ", "lowest ")
        )
        return names_action and ("student" in lowered_query)

    @classmethod
    def _heuristic_plan(cls, query: str) -> dict[str, Any]:
        lowered = query.casefold()
        rag = any(cue in lowered for cue in RAG_CUES)
        has_data_entity = any(cue in lowered for cue in SQL_CUES)
        has_database_action = any(cue in lowered for cue in SQL_ACTION_CUES)
        sql = has_data_entity and has_database_action
        analytics = any(cue in lowered for cue in ANALYTICS_CUES)

        if rag:
            intents = ["rag"]
        else:
            intents = []
        if sql:
            intents.append("sql")
        if analytics:
            intents.append("analytics")
        # A table plus policy explanation benefits from both the requested
        # records and a short statistical interpretation in the unified reply.
        if rag and sql and not analytics:
            intents.append("analytics")
        if not intents:
            intents = ["rag"]

        scope = "none"
        if "sql" in intents:
            if cls._is_aggregate_request(query):
                scope = "aggregate"
            elif cls._is_raw_student_request(lowered) or any(
                cue in lowered for cue in ("attendance", "exam", "grade", "marks", "cgpa")
            ):
                scope = "individual"
            else:
                # Department/course catalog queries do not expose student records.
                scope = "aggregate"

        sub_queries = {
            intent: query.strip() if intent in intents else ""
            for intent in INTENTS
        }
        return {
            "intents": intents,
            "sub_queries": sub_queries,
            "sql_scope": scope,
        }

    def _generate_plan(self, query: str, role: str) -> dict[str, Any]:
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=build_router_prompt(query, role=role),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ROUTER_RESPONSE_SCHEMA,
                "temperature": 0.0,
            },
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty structured route")
        decoded = json.loads(text)
        if not isinstance(decoded, dict):
            raise ValueError("Gemini route response must be a JSON object")
        return decoded

    @staticmethod
    def _safe_plan_shape(raw: dict[str, Any], query: str) -> dict[str, Any]:
        requested = raw.get("intents", [])
        intents: list[str] = []
        if isinstance(requested, list):
            for item in requested:
                if isinstance(item, str) and item in INTENTS and item not in intents:
                    intents.append(item)
        sub_queries_raw = raw.get("sub_queries")
        sub_queries = {
            intent: (
                str(sub_queries_raw.get(intent, "")).strip()
                if isinstance(sub_queries_raw, dict)
                else ""
            )
            for intent in INTENTS
        }
        scope = raw.get("sql_scope", "none")
        if scope not in {"none", "aggregate", "self", "individual"}:
            scope = "none"
        if not intents:
            return IntentRouter._heuristic_plan(query)
        for intent in intents:
            if not sub_queries[intent]:
                sub_queries[intent] = query.strip()
        for intent in INTENTS:
            if intent not in intents:
                sub_queries[intent] = ""
        return {"intents": intents, "sub_queries": sub_queries, "sql_scope": scope}

    def route(
        self,
        query: str,
        *,
        role: str = "Faculty / Administrator",
        student_id: int | None = None,
    ) -> RoutePlan:
        """Route one request; use Gemini JSON mode when configured, else heuristics."""
        if not query.strip():
            raise ValueError("query must not be empty")
        normalized_role = self._normalize_role(role)
        route_source = "gemini"

        if self.use_llm and (self.client is not None or self.api_key):
            try:
                raw = self._generate_plan(query, normalized_role)
                raw = self._safe_plan_shape(raw, query)
            except Exception:
                raw = self._heuristic_plan(query)
                route_source = "heuristic_fallback"
        else:
            raw = self._heuristic_plan(query)
            route_source = "heuristic"

        intents = list(raw["intents"])
        sub_queries = dict(raw["sub_queries"])
        sql_scope = raw.get("sql_scope", "none")
        blocked: list[str] = []
        access_message = None

        if normalized_role == "Student" and "sql" in intents:
            if sql_scope == "aggregate":
                pass
            elif sql_scope in {"self", "individual"} and student_id is not None:
                pass
            else:
                intents.remove("sql")
                sub_queries["sql"] = ""
                blocked.append("sql")
                access_message = (
                    "Raw individual student records require an authenticated student "
                    "identity. This session can use policy answers and aggregate summaries."
                )

        return RoutePlan(
            intents=intents,
            sub_queries=sub_queries,
            sql_scope=sql_scope,
            blocked_intents=blocked,
            access_message=access_message,
            route_source=route_source,
        )


def route_query(
    query: str,
    *,
    role: str = "Faculty / Administrator",
    student_id: int | None = None,
    router: IntentRouter | None = None,
) -> RoutePlan:
    """Convenience wrapper for routing from Streamlit or evaluation scripts."""
    return (router or IntentRouter()).route(
        query, role=role, student_id=student_id
    )
