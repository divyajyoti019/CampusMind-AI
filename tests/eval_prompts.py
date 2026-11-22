"""Offline-first 20-case router, SQL, and policy-citation evaluation harness.

Run offline with python tests/eval_prompts.py. Add --live to evaluate Gemini
routing/SQL/RAG calls with GEMINI_API_KEY and a locally built embedding index.
The live option can make multiple Gemini requests and may incur API cost.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from database.db import execute_readonly_query, initialize_database
from database.sample_data import seed_sample_data
from modules.intent_router import IntentRouter
from modules.rag_assistant import RAGAssistant
from modules.text_to_sql import TextToSQLAssistant
from utils.pdf_processor import extract_pdf_pages
from utils.security import validate_and_sanitize_sql


@dataclass(frozen=True)
class EvalCase:
    query: str
    intents: tuple[str, ...]
    sql: str | None = None
    citation_page: int | None = None
    citation_terms: tuple[str, ...] = ()


CASES = [
    EvalCase("What is the university minimum attendance requirement?", ("rag",), citation_page=1, citation_terms=("75%", "each registered course")),
    EvalCase("Show students with attendance below 75%.", ("sql",), sql="SELECT s.roll_no, s.full_name, a.attendance_pct FROM attendance a JOIN students s USING (student_id) WHERE a.attendance_pct < 75"),
    EvalCase("How many students are below the attendance threshold?", ("sql", "analytics"), sql="SELECT COUNT(DISTINCT student_id) AS low_attendance_students FROM attendance WHERE attendance_pct < 75"),
    EvalCase("Compare average attendance by department.", ("sql", "analytics"), sql="SELECT d.dept_name, AVG(a.attendance_pct) AS average_attendance FROM attendance a JOIN students s USING (student_id) JOIN departments d USING (dept_id) GROUP BY d.dept_id, d.dept_name"),
    EvalCase("Show students below 75% attendance and explain the university attendance policy.", ("sql", "rag", "analytics"), sql="SELECT s.roll_no, s.full_name, a.attendance_pct FROM attendance a JOIN students s USING (student_id) WHERE a.attendance_pct < 75", citation_page=1, citation_terms=("75%", "medical condonation")),
    EvalCase("How does medical condonation work for attendance?", ("rag",), citation_page=1, citation_terms=("65%", "seven calendar days", "Dean (Academic Affairs)")),
    EvalCase("Show five students with the highest average exam percentage.", ("sql", "analytics"), sql="SELECT s.roll_no, s.full_name, AVG(100.0 * e.marks_obtained / e.max_marks) AS average_exam_pct FROM exam_results e JOIN students s USING (student_id) GROUP BY s.student_id ORDER BY average_exam_pct DESC LIMIT 5"),
    EvalCase("What are the tuition payment and late-fee rules?", ("rag",), citation_page=3, citation_terms=("10th calendar day", "INR 100", "INR 1,500")),
    EvalCase("When are hostel quiet hours and visitor hours?", ("rag",), citation_page=4, citation_terms=("10:00 p.m. to 6:00 a.m.", "8:00 a.m. and 8:00 p.m.")),
    EvalCase("How many books can a student borrow from the library?", ("rag",), citation_page=4, citation_terms=("four books for 14 days",)),
    EvalCase("List courses offered by CSE in semester three.", ("sql",), sql="SELECT c.course_code, c.course_name FROM courses c JOIN departments d USING (dept_id) WHERE d.dept_code = 'CSE' AND c.semester = 3"),
    EvalCase("What is the average CGPA by department?", ("sql", "analytics"), sql="SELECT d.dept_name, AVG(s.cgpa) AS average_cgpa FROM students s JOIN departments d USING (dept_id) GROUP BY d.dept_id, d.dept_name"),
    EvalCase("Show the course-wise exam score distribution.", ("sql", "analytics"), sql="SELECT c.course_name, AVG(100.0 * e.marks_obtained / e.max_marks) AS average_score FROM exam_results e JOIN courses c USING (course_id) GROUP BY c.course_id, c.course_name"),
    EvalCase("Is attendance correlated with exam marks?", ("sql", "analytics"), sql="SELECT a.attendance_pct, 100.0 * e.marks_obtained / e.max_marks AS exam_pct FROM attendance a JOIN exam_results e ON e.student_id = a.student_id AND e.course_id = a.course_id"),
    EvalCase("Count courses in each department.", ("sql", "analytics"), sql="SELECT d.dept_name, COUNT(c.course_id) AS course_count FROM departments d LEFT JOIN courses c USING (dept_id) GROUP BY d.dept_id, d.dept_name"),
    EvalCase("Show ECE students in semester five.", ("sql",), sql="SELECT s.roll_no, s.full_name FROM students s JOIN departments d USING (dept_id) WHERE d.dept_code = 'ECE' AND s.semester = 5"),
    EvalCase("Explain the exam grade bands and passing mark.", ("rag",), citation_page=2, citation_terms=("Grade bands", "Below 35", "35%")),
    EvalCase("Explain the attendance policy and count students below its threshold.", ("rag", "sql", "analytics"), sql="SELECT COUNT(DISTINCT student_id) AS low_attendance_students FROM attendance WHERE attendance_pct < 75", citation_page=1, citation_terms=("75%", "each registered course")),
    EvalCase("What percentage of exam marks are students earning in each department?", ("sql", "analytics"), sql="SELECT d.dept_name, AVG(100.0 * e.marks_obtained / e.max_marks) AS average_exam_pct FROM exam_results e JOIN students s USING (student_id) JOIN departments d USING (dept_id) GROUP BY d.dept_id, d.dept_name", citation_page=2, citation_terms=("40 marks", "60 marks")),
    EvalCase("Summarize attendance patterns and show the department attendance chart.", ("sql", "analytics"), sql="SELECT d.dept_name, AVG(a.attendance_pct) AS average_attendance FROM attendance a JOIN students s USING (student_id) JOIN departments d USING (dept_id) GROUP BY d.dept_id, d.dept_name"),
]


_CITATION = re.compile(r"\[([^\],]+),\s*Page\s+(\d+)\]")


def _ensure_database() -> None:
    if not settings.database_path.exists():
        initialize_database(settings.database_path)
        seed_sample_data(settings.database_path)


def _offline_citation(case: EvalCase, pages: list[dict[str, Any]]) -> bool:
    if case.citation_page is None:
        return True
    page = next((item for item in pages if item["page"] == case.citation_page), None)
    if page is None:
        return False
    content = page["text"].casefold()
    return all(term.casefold() in content for term in case.citation_terms)


def _live_citation(case: EvalCase, assistant: RAGAssistant) -> bool:
    output = assistant.answer_with_sources(case.query, role="Faculty / Administrator")
    citations = _CITATION.findall(output["answer"])
    if not citations or output["answer"].startswith("The uploaded university documents"):
        return False
    source_pairs = {
        (
            str(chunk.get("source") or chunk.get("source_file") or ""),
            str(chunk.get("page") or chunk.get("page_number") or ""),
        )
        for chunk in output["sources"]
    }
    return all((source, page) in source_pairs for source, page in citations)


def run_evaluation(live: bool = False) -> dict[str, tuple[int, int]]:
    _ensure_database()
    router = IntentRouter(use_llm=live)
    sql_assistant = TextToSQLAssistant() if live else None
    rag_assistant = RAGAssistant() if live else None
    if live:
        if not settings.gemini_api_key:
            raise RuntimeError("--live requires GEMINI_API_KEY")
        if not settings.faiss_index_path.exists():
            rag_assistant.ingest_documents(settings.documents_dir)
    pages = extract_pdf_pages(settings.documents_dir / "University_Academic_Handbook.pdf")

    true_positives = 0
    predicted_count = 0
    sql_total = sql_success = 0
    rag_total = rag_success = 0

    for case in CASES:
        plan = router.route(case.query, role="Faculty / Administrator")
        predicted = set(plan.intents)
        expected = set(case.intents)
        true_positives += len(predicted & expected)
        predicted_count += len(predicted)

        if case.sql:
            sql_total += 1
            if live:
                sql_result = sql_assistant.query(case.query)
                sql_success += int(sql_result["error"] is None)
            else:
                try:
                    safe_sql = validate_and_sanitize_sql(case.sql)
                    execute_readonly_query(safe_sql, db_path=settings.database_path)
                    sql_success += 1
                except Exception:
                    pass

        if case.citation_page is not None:
            rag_total += 1
            if live:
                rag_success += int(_live_citation(case, rag_assistant))
            else:
                rag_success += int(_offline_citation(case, pages))

    return {
        "routing precision": (true_positives, predicted_count),
        "SQL execution success": (sql_success, sql_total),
        "RAG citation accuracy": (rag_success, rag_total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use Gemini and the actual RAG index (may make billable API calls).",
    )
    args = parser.parse_args()
    try:
        results = run_evaluation(live=args.live)
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print(f"CampusMind prompt evaluation · {len(CASES)} cases · {'live' if args.live else 'offline'} mode")
    print("-" * 66)
    print(f"{'Metric':<30} {'Score':>12} {'Passed / total':>20}")
    print("-" * 66)
    for label, (passed, total) in results.items():
        score = (passed / total * 100.0) if total else 0.0
        print(f"{label:<30} {score:>11.1f}% {passed:>9} / {total:<10}")
    print("-" * 66)
    print("Offline mode uses deterministic routing, canonical read-only SQL, and handbook-page checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
