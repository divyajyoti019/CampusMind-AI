"""Analytics KPI, Plotly figure, watchlist, and insight-grounding tests."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytest

from database.sample_data import seed_sample_data
from modules.analytics import (
    build_analytics_dashboard,
    build_attendance_exam_scatter_chart,
    build_course_score_distribution_chart,
    build_department_attendance_chart,
    compute_kpis,
    compute_summary_statistics,
    generate_executive_summary,
    get_low_attendance_watchlist,
    load_analytics_data,
    watchlist_to_csv,
)


class FakeGemini:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0
        self.models = self

    def generate_content(self, *, model: str, contents: str) -> Any:
        self.calls += 1
        assert "Computed summary statistics (JSON):" in contents
        assert '"total_students": 100' in contents
        return type("Response", (), {"text": self.response_text})()


@pytest.fixture
def sample_db(tmp_path):
    return seed_sample_data(tmp_path / "campusmind.db")


def test_core_kpis_match_independent_dataframe_calculations(sample_db) -> None:
    data = load_analytics_data(sample_db)
    expected_attendance = float(data.attendance["attendance_pct"].mean())
    expected_score = float(
        (100.0 * data.exam_results["marks_obtained"] / data.exam_results["max_marks"]).mean()
    )
    expected_low = data.attendance.loc[
        data.attendance["attendance_pct"] < 75, "student_id"
    ].nunique()

    kpis = compute_kpis(data=data)

    assert kpis == {
        "total_students": 100,
        "overall_avg_attendance_pct": round(expected_attendance, 2),
        "overall_avg_score_pct": round(expected_score, 2),
        "students_below_attendance_threshold": int(expected_low),
    }
    assert kpis["students_below_attendance_threshold"] == 25


def test_plotly_charts_have_expected_data_and_threshold_line(sample_db) -> None:
    data = load_analytics_data(sample_db)
    attendance = build_department_attendance_chart(data=data)
    course_scores = build_course_score_distribution_chart(data=data)
    scatter = build_attendance_exam_scatter_chart(data=data)

    assert isinstance(attendance, go.Figure)
    assert len(attendance.data) == 5
    assert any(shape.line.dash == "dash" and shape.y0 == 75 for shape in attendance.layout.shapes)

    assert isinstance(course_scores, go.Figure)
    assert len(course_scores.data) == 20
    assert all(trace.type == "box" for trace in course_scores.data)

    assert isinstance(scatter, go.Figure)
    point_count = sum(len(trace.x) for trace in scatter.data)
    assert point_count == len(data.exam_results)
    assert len(scatter.data) == 5


def test_low_attendance_watchlist_filters_and_csv_export(sample_db) -> None:
    watchlist = get_low_attendance_watchlist(sample_db)
    cse_watchlist = get_low_attendance_watchlist(sample_db, department="CSE")

    assert not watchlist.empty
    assert (watchlist["attendance_pct"] < 75).all()
    assert watchlist["roll_no"].notna().all()
    assert watchlist["course_code"].notna().all()
    assert watchlist["roll_no"].nunique() == 25
    assert set(cse_watchlist["dept_code"]) == {"CSE"}
    assert cse_watchlist["attendance_pct"].max() < 75
    csv = watchlist_to_csv(watchlist)
    assert isinstance(csv, bytes)
    assert b"roll_no,full_name" in csv


def test_ai_summary_receives_computed_json_and_dashboard_returns_outputs(sample_db) -> None:
    stats = compute_summary_statistics(sample_db)
    client = FakeGemini(
        "Course-level attendance shows a group below the stated threshold. "
        "Prioritize advising outreach and review course score distributions. "
        "This snapshot does not establish historical trends or causation."
    )
    summary = generate_executive_summary(stats, client=client)
    assert "Prioritize advising outreach" in summary
    assert client.calls == 1

    dashboard = build_analytics_dashboard(sample_db, client=client)
    assert dashboard["summary_error"] is None
    assert dashboard["executive_summary"]
    assert set(dashboard["figures"]) == {
        "department_attendance",
        "course_exam_scores",
        "attendance_exam_scatter",
    }
    assert isinstance(dashboard["low_attendance_watchlist"], pd.DataFrame)


def test_summary_rejects_numbers_not_in_computed_statistics(sample_db) -> None:
    stats = compute_summary_statistics(sample_db)
    client = FakeGemini("Attendance improved by 1000000% this year.")
    with pytest.raises(ValueError, match="unsupported numeric"):
        generate_executive_summary(stats, client=client)
