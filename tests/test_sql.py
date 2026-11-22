"""Read-only Text-to-SQL and SQL-injection boundary tests."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import modules.text_to_sql as text_to_sql_module
from database.sample_data import seed_sample_data
from modules.text_to_sql import TextToSQLAssistant
from utils.security import validate_and_sanitize_sql


class QueueGemini:
    def __init__(self, sql_responses: list[str]) -> None:
        self.sql_responses = list(sql_responses)
        self.calls = 0
        self.models = self

    def generate_content(self, *, model: str, contents: str) -> Any:
        self.calls += 1
        assert "SQLite schema:" in contents
        if not self.sql_responses:
            raise AssertionError("Unexpected Gemini generation call")
        return type("Response", (), {"text": self.sql_responses.pop(0)})()


@pytest.fixture
def sample_db(tmp_path):
    return seed_sample_data(tmp_path / "campusmind.db")


QUERY_CASES = [
    (
        "How many departments are there?",
        "SELECT COUNT(*) AS department_count FROM departments",
    ),
    (
        "List the department names and their codes.",
        "SELECT dept_name, dept_code FROM departments ORDER BY dept_name",
    ),
    (
        "How many students are in each department?",
        "SELECT d.dept_name, COUNT(s.student_id) AS student_count "
        "FROM departments d LEFT JOIN students s ON s.dept_id = d.dept_id "
        "GROUP BY d.dept_id, d.dept_name ORDER BY student_count DESC",
    ),
    (
        "Count students in the CSE department.",
        "SELECT COUNT(s.student_id) AS student_count FROM students s "
        "JOIN departments d ON d.dept_id = s.dept_id WHERE d.dept_code = 'CSE'",
    ),
    (
        "What is the average CGPA by department?",
        "SELECT d.dept_name, ROUND(AVG(s.cgpa), 2) AS average_cgpa "
        "FROM students s JOIN departments d ON d.dept_id = s.dept_id "
        "GROUP BY d.dept_id, d.dept_name ORDER BY average_cgpa DESC",
    ),
    (
        "Show average attendance by department.",
        "SELECT d.dept_name, ROUND(AVG(a.attendance_pct), 2) AS average_attendance "
        "FROM attendance a JOIN students s ON s.student_id = a.student_id "
        "JOIN departments d ON d.dept_id = s.dept_id "
        "GROUP BY d.dept_id, d.dept_name ORDER BY average_attendance DESC",
    ),
    (
        "Show students with attendance below 75 percent.",
        "SELECT s.roll_no, s.full_name, c.course_name, a.attendance_pct "
        "FROM attendance a JOIN students s ON s.student_id = a.student_id "
        "JOIN courses c ON c.course_id = a.course_id "
        "WHERE a.attendance_pct < 75 ORDER BY a.attendance_pct",
    ),
    (
        "How many unique students are below the attendance threshold?",
        "SELECT COUNT(DISTINCT student_id) AS low_attendance_students "
        "FROM attendance WHERE attendance_pct < 75",
    ),
    (
        "Count courses offered by each department.",
        "SELECT d.dept_name, COUNT(c.course_id) AS course_count "
        "FROM departments d LEFT JOIN courses c ON c.dept_id = d.dept_id "
        "GROUP BY d.dept_id, d.dept_name ORDER BY d.dept_name",
    ),
    (
        "List CSE courses in semester three.",
        "SELECT c.course_code, c.course_name, c.credits FROM courses c "
        "JOIN departments d ON d.dept_id = c.dept_id "
        "WHERE d.dept_code = 'CSE' AND c.semester = 3 ORDER BY c.course_code",
    ),
    (
        "Who are the five students with the highest CGPA?",
        "SELECT roll_no, full_name, cgpa FROM students ORDER BY cgpa DESC LIMIT 5",
    ),
    (
        "What is the average final exam percentage for each course?",
        "SELECT c.course_name, ROUND(AVG(100.0 * e.marks_obtained / e.max_marks), 2) "
        "AS average_final_pct FROM exam_results e "
        "JOIN courses c ON c.course_id = e.course_id WHERE e.exam_type = 'Final' "
        "GROUP BY c.course_id, c.course_name ORDER BY average_final_pct DESC",
    ),
    (
        "Show the top five students by average exam percentage.",
        "SELECT s.roll_no, s.full_name, "
        "ROUND(AVG(100.0 * e.marks_obtained / e.max_marks), 2) AS avg_exam_pct "
        "FROM exam_results e JOIN students s ON s.student_id = e.student_id "
        "GROUP BY s.student_id, s.roll_no, s.full_name "
        "ORDER BY avg_exam_pct DESC LIMIT 5",
    ),
    (
        "How many students received each grade in the midterm?",
        "SELECT grade, COUNT(*) AS result_count FROM exam_results "
        "WHERE exam_type = 'Midterm' GROUP BY grade ORDER BY grade",
    ),
    (
        "How many students are in semester one in each department?",
        "SELECT d.dept_name, COUNT(s.student_id) AS student_count "
        "FROM students s JOIN departments d ON d.dept_id = s.dept_id "
        "WHERE s.semester = 1 GROUP BY d.dept_id, d.dept_name ORDER BY d.dept_name",
    ),
]


@pytest.mark.parametrize("question, generated_sql", QUERY_CASES)
def test_natural_language_queries_execute_as_read_only_selects(
    sample_db, question: str, generated_sql: str
) -> None:
    client = QueueGemini([generated_sql])
    assistant = TextToSQLAssistant(db_path=sample_db, client=client)

    result = assistant.query(question)

    assert result["error"] is None
    assert isinstance(result["dataframe"], pd.DataFrame)
    assert "LIMIT" in result["sql"].upper()
    assert "returned" in result["explanation"]
    assert client.calls == 1


MALICIOUS_SQL = [
    "DROP TABLE students",
    "UPDATE students SET cgpa = 0",
    "SELECT student_id FROM students; DROP TABLE students",
    "PRAGMA table_info(students)",
    "SELECT name FROM sqlite_master WHERE type = 'table'",
]


@pytest.mark.parametrize("generated_sql", MALICIOUS_SQL)
def test_malicious_sql_is_blocked_before_database_execution(
    sample_db, generated_sql: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = QueueGemini([generated_sql, generated_sql])
    assistant = TextToSQLAssistant(db_path=sample_db, client=client)
    execution_calls: list[str] = []

    def record_execution(sql: str, **kwargs: Any) -> pd.DataFrame:
        execution_calls.append(sql)
        raise AssertionError("Invalid SQL reached SQLite execution")

    monkeypatch.setattr(text_to_sql_module, "execute_readonly_query", record_execution)
    result = assistant.query("Please run this database operation")

    assert result["error"]
    assert result["dataframe"].empty
    assert execution_calls == []
    assert client.calls == 2  # one generation plus one permitted correction


def test_security_injects_default_limit_and_caps_large_limits() -> None:
    assert validate_and_sanitize_sql("SELECT dept_name FROM departments").endswith(
        "LIMIT 100"
    )
    capped = validate_and_sanitize_sql("SELECT dept_name FROM departments LIMIT 500")
    assert capped.endswith("LIMIT 100")


def test_validation_failure_gets_one_correction_then_executes(sample_db) -> None:
    client = QueueGemini(
        [
            "SELECT full_name FROM students; DROP TABLE students",
            "SELECT COUNT(*) AS student_count FROM students",
        ]
    )
    assistant = TextToSQLAssistant(db_path=sample_db, client=client)

    result = assistant.query("How many students are registered?")

    assert result["error"] is None
    assert result["dataframe"].iloc[0]["student_count"] == 100
    assert client.calls == 2


def test_student_without_identity_can_run_aggregate_summary(sample_db) -> None:
    client = QueueGemini(["SELECT COUNT(*) AS student_count FROM students"])
    result = TextToSQLAssistant(db_path=sample_db, client=client).query(
        "How many students are registered?", role="Student"
    )
    assert result["error"] is None
    assert result["dataframe"].iloc[0]["student_count"] == 100


def test_student_raw_records_require_scope_and_are_scoped(sample_db) -> None:
    client = QueueGemini(
        [
            "SELECT student_id, roll_no, full_name FROM students",
            "SELECT student_id, roll_no, full_name FROM students",
        ]
    )
    result = TextToSQLAssistant(db_path=sample_db, client=client).query(
        "Show my student details", role="Student"
    )
    assert result["error"]
    assert result["dataframe"].empty

    scoped_client = QueueGemini(
        ["SELECT student_id, roll_no, full_name FROM students"]
    )
    scoped_result = TextToSQLAssistant(db_path=sample_db, client=scoped_client).query(
        "Show my student details", role="Student", student_id=1
    )
    assert scoped_result["error"] is None
    assert scoped_result["dataframe"]["student_id"].tolist() == [1]


def test_execution_error_gets_one_correction_attempt(sample_db, monkeypatch) -> None:
    client = QueueGemini(
        [
            "SELECT COUNT(*) AS student_count FROM students",
            "SELECT COUNT(*) AS student_count FROM students",
        ]
    )
    assistant = TextToSQLAssistant(db_path=sample_db, client=client)
    execution_count = 0

    def fail_once_then_return(sql: str, **kwargs: Any) -> pd.DataFrame:
        nonlocal execution_count
        execution_count += 1
        if execution_count == 1:
            raise RuntimeError("simulated SQLite execution error")
        return pd.DataFrame({"student_count": [100]})

    monkeypatch.setattr(text_to_sql_module, "execute_readonly_query", fail_once_then_return)
    result = assistant.query("How many students are registered?")

    assert result["error"] is None
    assert result["dataframe"].iloc[0]["student_count"] == 100
    assert execution_count == 2
    assert client.calls == 2
