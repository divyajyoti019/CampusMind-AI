"""Statistical KPIs, Plotly figures, low-attendance watchlist, and AI narrative."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings
from database.db import DEFAULT_DB_PATH, get_connection
from prompts.analytics_prompt import build_analytics_prompt


@dataclass
class AnalyticsData:
    """Read-only data frames needed for dashboard metrics and charts."""

    departments: pd.DataFrame
    students: pd.DataFrame
    courses: pd.DataFrame
    attendance: pd.DataFrame
    exam_results: pd.DataFrame


def load_analytics_data(db_path: str | Path = DEFAULT_DB_PATH) -> AnalyticsData:
    """Load complete table snapshots from a read-only SQLite connection."""
    connection = get_connection(db_path, read_only=True)
    try:
        return AnalyticsData(
            departments=pd.read_sql_query("SELECT * FROM departments", connection),
            students=pd.read_sql_query("SELECT * FROM students", connection),
            courses=pd.read_sql_query("SELECT * FROM courses", connection),
            attendance=pd.read_sql_query("SELECT * FROM attendance", connection),
            exam_results=pd.read_sql_query("SELECT * FROM exam_results", connection),
        )
    finally:
        connection.close()


def _attendance_records(data: AnalyticsData) -> pd.DataFrame:
    return (
        data.attendance.merge(
            data.students[["student_id", "dept_id", "semester"]],
            on="student_id",
            how="left",
        )
        .merge(
            data.departments[["dept_id", "dept_name", "dept_code"]],
            on="dept_id",
            how="left",
        )
    )


def _exam_records(data: AnalyticsData) -> pd.DataFrame:
    if data.exam_results.empty:
        empty = data.exam_results.copy()
        empty["score_pct"] = pd.Series(dtype=float)
        empty["course_name"] = pd.Series(dtype=str)
        empty["course_code"] = pd.Series(dtype=str)
        empty["dept_name"] = pd.Series(dtype=str)
        empty["dept_code"] = pd.Series(dtype=str)
        return empty
    records = (
        data.exam_results.merge(
            data.courses[["course_id", "course_code", "course_name", "dept_id"]],
            on="course_id",
            how="left",
        )
        .merge(
            data.departments[["dept_id", "dept_name", "dept_code"]],
            on="dept_id",
            how="left",
        )
    )
    records["score_pct"] = (
        100.0 * records["marks_obtained"] / records["max_marks"]
    )
    return records


def _attendance_exam_records(data: AnalyticsData) -> pd.DataFrame:
    attendance = _attendance_records(data)
    exams = _exam_records(data)
    if attendance.empty or exams.empty:
        return pd.DataFrame(
            columns=["student_id", "course_id", "attendance_pct", "marks_obtained", "max_marks", "score_pct", "dept_name"]
        )
    return attendance.merge(
        exams[
            [
                "student_id",
                "course_id",
                "exam_type",
                "marks_obtained",
                "max_marks",
                "score_pct",
            ]
        ],
        on=["student_id", "course_id"],
        how="inner",
        suffixes=("_attendance", "_exam"),
    )


def compute_kpis(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    attendance_threshold_pct: float = settings.low_attendance_threshold_pct,
    data: AnalyticsData | None = None,
) -> dict[str, int | float]:
    """Compute student count, mean attendance/score percentages, and low count.

    Attendance and score means are averages of the recorded course/assessment
    percentages; the low-attendance KPI counts distinct students with at least
    one course below the threshold.
    """
    if not 0 <= attendance_threshold_pct <= 100:
        raise ValueError("attendance_threshold_pct must be between 0 and 100")
    data = data or load_analytics_data(db_path)
    attendance_records = _attendance_records(data)
    exam_records = _exam_records(data)

    attendance_mean = (
        float(attendance_records["attendance_pct"].mean())
        if not attendance_records.empty
        else 0.0
    )
    score_mean = (
        float(exam_records["score_pct"].mean()) if not exam_records.empty else 0.0
    )
    low_ids = (
        attendance_records.loc[
            attendance_records["attendance_pct"] < attendance_threshold_pct,
            "student_id",
        ].nunique()
        if not attendance_records.empty
        else 0
    )
    return {
        "total_students": int(len(data.students)),
        "overall_avg_attendance_pct": round(attendance_mean, 2),
        "overall_avg_score_pct": round(score_mean, 2),
        "students_below_attendance_threshold": int(low_ids),
    }


def build_department_attendance_chart(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    attendance_threshold_pct: float = settings.low_attendance_threshold_pct,
    data: AnalyticsData | None = None,
) -> go.Figure:
    """Build department-wise average attendance with the minimum line."""
    data = data or load_analytics_data(db_path)
    records = _attendance_records(data)
    if records.empty:
        grouped = pd.DataFrame(columns=["dept_name", "attendance_pct"])
    else:
        grouped = (
            records.groupby(["dept_id", "dept_name"], as_index=False)["attendance_pct"]
            .mean()
            .sort_values("attendance_pct", ascending=False)
        )
    fig = px.bar(
        grouped,
        x="dept_name",
        y="attendance_pct",
        color="dept_name",
        labels={"dept_name": "Department", "attendance_pct": "Average attendance (%)"},
        title="Department-wise Average Attendance",
        hover_data={"attendance_pct": ":.2f"},
    )
    fig.add_hline(
        y=attendance_threshold_pct,
        line_dash="dash",
        line_color="#C62828",
        annotation_text=f"{attendance_threshold_pct:g}% minimum",
        annotation_position="top left",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    return fig


def build_course_score_distribution_chart(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    data: AnalyticsData | None = None,
) -> go.Figure:
    """Build a course-wise box plot of exam score percentages."""
    data = data or load_analytics_data(db_path)
    records = _exam_records(data)
    fig = px.box(
        records,
        x="course_name",
        y="score_pct",
        color="course_name",
        points="outliers",
        labels={"course_name": "Course", "score_pct": "Exam score (%)"},
        title="Course-wise Exam Score Distribution",
        hover_data=["course_code", "exam_type"],
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 100], xaxis_tickangle=-35)
    return fig


def build_attendance_exam_scatter_chart(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    data: AnalyticsData | None = None,
) -> go.Figure:
    """Build attendance-versus-exam-score scatter colored by department."""
    data = data or load_analytics_data(db_path)
    records = _attendance_exam_records(data)
    fig = px.scatter(
        records,
        x="attendance_pct",
        y="score_pct",
        color="dept_name",
        hover_data=["course_id", "exam_type", "marks_obtained", "max_marks"],
        labels={
            "attendance_pct": "Attendance (%)",
            "score_pct": "Exam marks (%)",
            "dept_name": "Department",
        },
        title="Attendance vs. Exam Marks",
        opacity=0.75,
    )
    fig.update_layout(xaxis_range=[0, 100], yaxis_range=[0, 100])
    return fig


def get_low_attendance_watchlist(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    attendance_threshold_pct: float = settings.low_attendance_threshold_pct,
    department: str | None = None,
    semester: int | None = None,
    course_code: str | None = None,
    data: AnalyticsData | None = None,
) -> pd.DataFrame:
    """Return one row per below-threshold student-course record, with filters."""
    if not 0 <= attendance_threshold_pct <= 100:
        raise ValueError("attendance_threshold_pct must be between 0 and 100")
    data = data or load_analytics_data(db_path)
    records = _attendance_records(data).merge(
        data.students[["student_id", "roll_no", "full_name", "email", "semester"]],
        on=["student_id", "semester"],
        how="left",
        suffixes=("", "_student"),
    ).merge(
        data.courses[["course_id", "course_code", "course_name"]],
        on="course_id",
        how="left",
    )
    watchlist = records.loc[
        records["attendance_pct"] < attendance_threshold_pct
    ].copy()
    if department:
        target = department.strip().casefold()
        watchlist = watchlist.loc[
            (watchlist["dept_name"].str.casefold() == target)
            | (watchlist["dept_code"].str.casefold() == target)
        ]
    if semester is not None:
        watchlist = watchlist.loc[watchlist["semester"] == semester]
    if course_code:
        watchlist = watchlist.loc[
            watchlist["course_code"].str.casefold() == course_code.strip().casefold()
        ]

    columns = [
        "roll_no",
        "full_name",
        "email",
        "dept_name",
        "dept_code",
        "semester",
        "course_code",
        "course_name",
        "classes_held",
        "classes_attended",
        "attendance_pct",
    ]
    return (
        watchlist[columns]
        .sort_values(["attendance_pct", "dept_name", "roll_no", "course_code"])
        .reset_index(drop=True)
    )


def watchlist_to_csv(watchlist: pd.DataFrame) -> bytes:
    """Encode a watchlist DataFrame for a Streamlit CSV download button."""
    return watchlist.to_csv(index=False).encode("utf-8")


def compute_summary_statistics(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    attendance_threshold_pct: float = settings.low_attendance_threshold_pct,
    data: AnalyticsData | None = None,
) -> dict[str, Any]:
    """Return JSON-safe KPI, department, course, and relationship summaries."""
    data = data or load_analytics_data(db_path)
    kpis = compute_kpis(
        attendance_threshold_pct=attendance_threshold_pct, data=data
    )
    attendance = _attendance_records(data)
    exams = _exam_records(data)
    paired = _attendance_exam_records(data)

    if attendance.empty:
        department_stats: list[dict[str, Any]] = []
        low_by_department: list[dict[str, Any]] = []
    else:
        department_stats = [
            {
                "department": row.dept_name,
                "average_attendance_pct": round(float(row.attendance_pct), 2),
            }
            for row in (
                attendance.groupby(["dept_id", "dept_name"], as_index=False)
                ["attendance_pct"].mean()
                .sort_values("dept_name")
                .itertuples(index=False)
            )
        ]
        low_rows = attendance.loc[
            attendance["attendance_pct"] < attendance_threshold_pct
        ]
        low_by_department = [
            {
                "department": row.dept_name,
                "students_below_threshold": int(row.student_count),
            }
            for row in (
                low_rows.groupby(["dept_id", "dept_name"])["student_id"]
                .nunique()
                .reset_index(name="student_count")
                .sort_values("dept_name")
                .itertuples(index=False)
            )
        ]

    if exams.empty:
        course_stats: list[dict[str, Any]] = []
    else:
        course_stats = []
        for (course_code, course_name), group in exams.groupby(
            ["course_code", "course_name"], sort=True
        ):
            scores = group["score_pct"].to_numpy(dtype=float)
            course_stats.append(
                {
                    "course_code": str(course_code),
                    "course_name": str(course_name),
                    "assessment_count": int(len(scores)),
                    "mean_score_pct": round(float(np.mean(scores)), 2),
                    "median_score_pct": round(float(np.median(scores)), 2),
                    "p25_score_pct": round(float(np.percentile(scores, 25)), 2),
                    "p75_score_pct": round(float(np.percentile(scores, 75)), 2),
                }
            )

    correlation = None
    if len(paired) >= 2:
        candidate = float(
            np.corrcoef(paired["attendance_pct"], paired["score_pct"])[0, 1]
        )
        if np.isfinite(candidate):
            correlation = round(candidate, 3)

    return {
        "attendance_threshold_pct": float(attendance_threshold_pct),
        "kpis": kpis,
        "department_attendance": department_stats,
        "low_attendance_by_department": low_by_department,
        "course_exam_score_distribution": course_stats,
        "attendance_exam_score_correlation": correlation,
        "paired_attendance_exam_record_count": int(len(paired)),
    }


_NUMBER_PATTERN = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def _numeric_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, bool) or value is None:
        return values
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value):
            values.add(str(float(value)))
        return values
    if isinstance(value, dict):
        for nested in value.values():
            values.update(_numeric_values(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            values.update(_numeric_values(nested))
    return values


def _contains_only_supported_numbers(text: str, summary_statistics: dict[str, Any]) -> bool:
    allowed = {float(value) for value in _numeric_values(summary_statistics)}
    for token in _NUMBER_PATTERN.findall(text):
        try:
            number = float(token.replace(",", ""))
        except ValueError:
            return False
        if number not in allowed:
            return False
    return True


def generate_executive_summary(
    summary_statistics: dict[str, Any],
    *,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Ask Gemini for a concise insight narrative and reject unsupported numbers."""
    if client is None:
        key = api_key or settings.gemini_api_key
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from google import genai

        client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model or settings.gemini_model,
        contents=build_analytics_prompt(summary_statistics),
    )
    narrative = (getattr(response, "text", None) or "").strip()
    if not narrative:
        raise ValueError("Gemini returned an empty analytics summary")
    if not _contains_only_supported_numbers(narrative, summary_statistics):
        raise ValueError("The generated summary included unsupported numeric values")
    return narrative


def build_analytics_dashboard(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    attendance_threshold_pct: float = settings.low_attendance_threshold_pct,
    department: str | None = None,
    semester: int | None = None,
    course_code: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Build all analytics outputs from one consistent database snapshot."""
    data = load_analytics_data(db_path)
    summary_statistics = compute_summary_statistics(
        attendance_threshold_pct=attendance_threshold_pct, data=data
    )
    summary = ""
    summary_error = None
    try:
        summary = generate_executive_summary(
            summary_statistics,
            client=client,
        )
    except (RuntimeError, ValueError) as exc:
        summary_error = str(exc)

    return {
        "kpis": summary_statistics["kpis"],
        "summary_statistics": summary_statistics,
        "executive_summary": summary,
        "summary_error": summary_error,
        "figures": {
            "department_attendance": build_department_attendance_chart(
                attendance_threshold_pct=attendance_threshold_pct, data=data
            ),
            "course_exam_scores": build_course_score_distribution_chart(data=data),
            "attendance_exam_scatter": build_attendance_exam_scatter_chart(data=data),
        },
        "low_attendance_watchlist": get_low_attendance_watchlist(
            attendance_threshold_pct=attendance_threshold_pct,
            department=department,
            semester=semester,
            course_code=course_code,
            data=data,
        ),
    }
