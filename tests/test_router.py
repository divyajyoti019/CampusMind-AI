"""Structured routing and role-based access tests."""

from __future__ import annotations

from typing import Any

from modules.intent_router import IntentRouter
from prompts.router_prompt import ROUTER_RESPONSE_SCHEMA


class FakeRouterClient:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.models = self
        self.call = None

    def generate_content(self, **kwargs: Any) -> Any:
        self.call = kwargs
        return type("Response", (), {"parsed": self.plan, "text": ""})()


def test_structured_compound_route_uses_requested_modules_and_decomposition() -> None:
    client = FakeRouterClient(
        {
            "intents": ["sql", "rag", "analytics"],
            "sub_queries": {
                "sql": "List students with course attendance below 75%.",
                "rag": "Summarize the official attendance rule and condonation process.",
                "analytics": "Count affected students and summarize attendance statistics.",
            },
            "sql_scope": "individual",
        }
    )
    router = IntentRouter(client=client)

    plan = router.route(
        "Show students below 75% attendance and explain the policy.",
        role="Faculty / Administrator",
    )

    assert plan.intents == ["sql", "rag", "analytics"]
    assert plan.sub_queries["rag"].startswith("Summarize")
    assert plan.blocked_intents == []
    assert client.call["config"]["response_mime_type"] == "application/json"
    assert client.call["config"]["response_json_schema"] == ROUTER_RESPONSE_SCHEMA


def test_heuristic_fallback_routes_compound_policy_and_data_questions() -> None:
    router = IntentRouter(use_llm=False)
    plan = router.route(
        "Show students whose attendance is below 75% and explain the university attendance policy.",
        role="Faculty / Administrator",
    )
    assert set(plan.intents) == {"sql", "rag", "analytics"}
    assert plan.route_source == "heuristic"


def test_student_raw_records_are_blocked_but_aggregate_sql_is_allowed() -> None:
    router = IntentRouter(use_llm=False)
    raw = router.route("Show students below 75% attendance.", role="Student")
    aggregate = router.route(
        "How many students have attendance below 75%?", role="Student"
    )
    self_query = router.route(
        "Show my attendance record.", role="Student", student_id=17
    )

    assert "sql" not in raw.intents
    assert raw.blocked_intents == ["sql"]
    assert "sql" in aggregate.intents
    assert "sql" in self_query.intents


def test_policy_only_query_does_not_route_to_sql() -> None:
    plan = IntentRouter(use_llm=False).route(
        "What does the attendance policy say about medical condonation?"
    )
    assert plan.intents == ["rag"]
