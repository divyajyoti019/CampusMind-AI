"""CampusMind AI - unified university assistant and analytics dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import settings
from database.db import get_connection, get_schema_metadata, initialize_database
from database.sample_data import seed_sample_data
from modules.analytics import (
    build_analytics_dashboard,
    build_attendance_exam_scatter_chart,
    build_course_score_distribution_chart,
    build_department_attendance_chart,
    compute_summary_statistics,
    generate_executive_summary,
    get_low_attendance_watchlist,
    load_analytics_data,
    watchlist_to_csv,
)
from modules.intent_router import IntentRouter
from modules.rag_assistant import RAGAssistant
from modules.text_to_sql import TextToSQLAssistant


SAMPLE_PROMPTS = (
    "Show students whose attendance is below 75% and explain the attendance policy.",
    "Compare average exam scores by department and show the course score distribution.",
    "What are the fee deadlines and how many students are in each department?",
)
_CITATION_RE = re.compile(r"\[([^\],]+),\s*Page\s+(\d+)\]")

st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_router() -> IntentRouter:
    return IntentRouter()


@st.cache_resource
def get_rag_assistant() -> RAGAssistant:
    return RAGAssistant()


@st.cache_resource
def get_sql_assistant() -> TextToSQLAssistant:
    return TextToSQLAssistant()


def _ensure_database() -> None:
    if not settings.database_path.exists():
        initialize_database(settings.database_path)
        seed_sample_data(settings.database_path)


def _database_health() -> tuple[bool, str]:
    try:
        _ensure_database()
        metadata = get_schema_metadata(settings.database_path)
        if not metadata:
            return False, "SQLite file exists, but the CampusMind schema is missing."
        connection = get_connection(settings.database_path, read_only=True)
        try:
            count = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        finally:
            connection.close()
        return True, f"SQLite healthy · {count:,} students · {len(metadata)} tables"
    except Exception as exc:
        return False, f"Database unavailable: {exc}"


def _persist_uploads(uploaded_files: list[Any]) -> None:
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name).name
        if not safe_name.lower().endswith(".pdf"):
            continue
        destination = settings.documents_dir / safe_name
        payload = uploaded.getvalue()
        if not destination.exists() or destination.read_bytes() != payload:
            destination.write_bytes(payload)


def _render_sidebar() -> tuple[str, str | None]:
    with st.sidebar:
        st.title("CampusMind AI")
        st.caption("University assistance · records · analytics")
        role = st.selectbox(
            "Active role",
            ["Student", "Faculty / Administrator"],
            key="active_role",
        )
        if role == "Student":
            st.caption("Raw SQL records require trusted student identity; aggregate summaries remain available.")

        st.divider()
        st.subheader("Policy knowledge base")
        uploaded_files = st.file_uploader(
            "Upload university policy PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="policy_pdf_upload",
        )
        _persist_uploads(uploaded_files or [])
        documents = sorted(settings.documents_dir.glob("*.pdf"))
        st.caption(f"{len(documents)} PDF document(s) available")
        if st.button("Re-index knowledge base", width="stretch"):
            if not documents:
                st.warning("Upload a PDF or add one under data/documents first.")
            else:
                try:
                    with st.spinner("Extracting pages and building the local FAISS index..."):
                        count = get_rag_assistant().ingest_documents(settings.documents_dir)
                    st.success(f"Indexed {count:,} text chunk(s).")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
        ready = settings.faiss_index_path.is_file() and settings.chunk_metadata_path.is_file()
        st.caption("Index ready" if ready else "Index not built yet")

        st.divider()
        st.subheader("Try a sample prompt")
        selected_prompt = None
        for index, prompt in enumerate(SAMPLE_PROMPTS):
            if st.button(prompt, key=f"sample_prompt_{index}", width="stretch"):
                selected_prompt = prompt

        st.divider()
        healthy, status = _database_health()
        st.success(status) if healthy else st.error(status)
        st.caption(
            "Gemini API key configured"
            if settings.gemini_api_key
            else "Gemini API key missing · add it to .env for live AI responses"
        )
    return role, selected_prompt


def _run_query(query: str, role: str) -> dict[str, Any]:
    plan = get_router().route(query, role=role)
    result: dict[str, Any] = {
        "plan": plan.to_dict(),
        "rag": None,
        "sql": None,
        "analytics": None,
        "narrative": [],
    }
    if plan.access_message:
        result["narrative"].append(plan.access_message)

    for intent in plan.intents:
        sub_query = plan.sub_queries.get(intent) or query
        try:
            if intent == "rag":
                rag_result = get_rag_assistant().answer_with_sources(sub_query, role=role)
                result["rag"] = rag_result
                result["narrative"].append(rag_result["answer"])
            elif intent == "sql":
                sql_result = get_sql_assistant().query(
                    sub_query,
                    role=role,
                    student_id=st.session_state.get("trusted_student_id"),
                )
                result["sql"] = sql_result
                if sql_result["error"]:
                    result["narrative"].append(
                        f"Database query could not be completed: {sql_result['error']}"
                    )
                else:
                    result["narrative"].append(sql_result["explanation"])
            elif intent == "analytics":
                stats = compute_summary_statistics(settings.database_path)
                summary, summary_error = "", None
                try:
                    summary = generate_executive_summary(stats)
                except Exception as exc:
                    summary_error = str(exc)
                lowered = sub_query.casefold()
                if any(word in lowered for word in ("exam", "marks", "score", "grade")):
                    figure = build_course_score_distribution_chart(settings.database_path)
                elif "correlation" in lowered or "relationship" in lowered:
                    figure = build_attendance_exam_scatter_chart(settings.database_path)
                else:
                    figure = build_department_attendance_chart(settings.database_path)
                result["analytics"] = {
                    "summary_statistics": stats,
                    "summary": summary,
                    "summary_error": summary_error,
                    "figure": figure,
                }
                result["narrative"].append(
                    summary or f"Computed academic KPIs: {stats['kpis']}"
                )
        except Exception as exc:
            result[intent] = {"error": str(exc)}
            result["narrative"].append(f"{intent.upper()} module error: {exc}")
    return result


def _render_citations(rag_result: dict[str, Any]) -> None:
    answer = rag_result.get("answer", "")
    citations = set(_CITATION_RE.findall(answer))
    sources = rag_result.get("sources", [])
    if citations:
        st.markdown("**Document citations**")
    for source in sources:
        reference = (
            str(source.get("source") or source.get("source_file") or ""),
            str(source.get("page") or source.get("page_number") or ""),
        )
        if citations and reference not in citations:
            continue
        with st.expander(f"📄 {reference[0]}, Page {reference[1]}"):
            st.write(source.get("text", ""))
            if source.get("score") is not None:
                st.caption(f"Retrieval similarity: {source['score']:.3f}")


def _render_chat_result(result: dict[str, Any]) -> None:
    plan = result.get("plan", {})
    selected = plan.get("intents", [])
    blocked = plan.get("blocked_intents") or []
    badges = " · ".join(f"**{item.upper()}**" for item in selected) or "**NO MODULE**"
    if blocked:
        badges += " · " + " · ".join(f"**{item.upper()} BLOCKED**" for item in blocked)
    st.markdown(f"**Routed to:** {badges}")
    if plan.get("route_source"):
        st.caption(f"Router: {plan['route_source']}")
    for message in result.get("narrative", []):
        st.markdown(message)

    if isinstance(result.get("rag"), dict) and result["rag"].get("sources"):
        _render_citations(result["rag"])

    sql_result = result.get("sql")
    if isinstance(sql_result, dict):
        with st.expander("SQL query and results", expanded=not bool(sql_result.get("error"))):
            st.code(sql_result.get("sql", ""), language="sql")
            frame = sql_result.get("dataframe")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.dataframe(frame, width="stretch", hide_index=True)
            elif sql_result.get("error"):
                st.error(sql_result["error"])
            else:
                st.info("The query returned no rows.")

    analytics_result = result.get("analytics")
    if isinstance(analytics_result, dict):
        with st.expander("Analytics chart", expanded=True):
            st.plotly_chart(
                analytics_result["figure"],
                width="stretch",
                key=f"chat_analytics_{id(analytics_result)}",
            )


def _render_assistant_tab(role: str) -> None:
    st.header("Unified AI Assistant")
    st.caption("Ask one question or combine policy, database, and analytics requests.")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_chat_result(message["result"])

    pending = st.session_state.pop("pending_query", None)
    typed = st.chat_input("Ask about university policies, records, or trends")
    current_query = pending or typed
    if current_query:
        st.session_state.chat_messages.append({"role": "user", "content": current_query})
        with st.chat_message("user"):
            st.markdown(current_query)
        with st.chat_message("assistant"):
            with st.spinner("Routing your request and gathering evidence..."):
                result = _run_query(current_query, role)
            _render_chat_result(result)
        st.session_state.chat_messages.append({"role": "assistant", "result": result})


def _render_kpis(kpis: dict[str, int | float]) -> None:
    columns = st.columns(4)
    columns[0].metric("Total students", f"{kpis['total_students']:,}")
    columns[1].metric("Average attendance", f"{kpis['overall_avg_attendance_pct']:.2f}%")
    columns[2].metric("Average exam score", f"{kpis['overall_avg_score_pct']:.2f}%")
    columns[3].metric(
        "Students below 75%",
        f"{kpis['students_below_attendance_threshold']:,}",
    )


def _render_analytics_tab(role: str) -> None:
    st.header("Academic Analytics Dashboard")
    if st.button("Refresh dashboard", key="refresh_analytics"):
        st.session_state.pop("analytics_dashboard", None)
    if "analytics_dashboard" not in st.session_state:
        with st.spinner("Computing university KPIs and charts..."):
            st.session_state.analytics_dashboard = build_analytics_dashboard(
                settings.database_path
            )
    dashboard = st.session_state.analytics_dashboard
    _render_kpis(dashboard["kpis"])

    st.plotly_chart(
        dashboard["figures"]["department_attendance"],
        width="stretch",
        key="dashboard_department_attendance",
    )
    st.plotly_chart(
        dashboard["figures"]["course_exam_scores"],
        width="stretch",
        key="dashboard_course_scores",
    )
    st.plotly_chart(
        dashboard["figures"]["attendance_exam_scatter"],
        width="stretch",
        key="dashboard_attendance_exam_scatter",
    )

    st.subheader("AI insight brief")
    if dashboard["executive_summary"]:
        st.info(dashboard["executive_summary"])
    elif dashboard["summary_error"]:
        st.caption(dashboard["summary_error"])

    if role == "Faculty / Administrator":
        st.subheader("Low-attendance student watchlist")
        data = load_analytics_data(settings.database_path)
        departments = ["All departments"] + sorted(
            data.departments["dept_name"].dropna().astype(str).unique().tolist()
        )
        course_codes = ["All courses"] + sorted(
            data.courses["course_code"].dropna().astype(str).unique().tolist()
        )
        filters = st.columns(3)
        department = filters[0].selectbox(
            "Department", departments, key="watchlist_department"
        )
        semester = filters[1].selectbox(
            "Semester", ["All semesters", *range(1, 9)], key="watchlist_semester"
        )
        course = filters[2].selectbox("Course", course_codes, key="watchlist_course")
        watchlist = get_low_attendance_watchlist(
            settings.database_path,
            department=None if department == "All departments" else department,
            semester=None if semester == "All semesters" else semester,
            course_code=None if course == "All courses" else course,
            data=data,
        )
        st.dataframe(watchlist, width="stretch", hide_index=True)
        st.download_button(
            "Download watchlist CSV",
            data=watchlist_to_csv(watchlist),
            file_name="campusmind_low_attendance_watchlist.csv",
            mime="text/csv",
            disabled=watchlist.empty,
        )
    else:
        st.caption("The detailed student watchlist and CSV export are for Faculty / Administrator.")


def main() -> None:
    role, selected_prompt = _render_sidebar()
    if selected_prompt:
        st.session_state.pending_query = selected_prompt
    st.title(f"{settings.app_icon} {settings.app_title}")
    assistant_tab, analytics_tab = st.tabs(
        ["Unified AI Assistant", "Analytics Dashboard"]
    )
    with assistant_tab:
        _render_assistant_tab(role)
    with analytics_tab:
        _render_analytics_tab(role)


if __name__ == "__main__":
    main()
