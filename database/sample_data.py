"""Create deterministic, realistic synthetic CampusMind university records."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

try:  # Supports both ``python -m database.sample_data`` and direct execution.
    from .db import DEFAULT_DB_PATH, initialize_database
except ImportError:  # pragma: no cover - exercised only when run as a script
    from db import DEFAULT_DB_PATH, initialize_database


RANDOM_SEED = 20261122

DEPARTMENTS = [
    ("Computer Science and Engineering", "CSE", "Dr. Ananya Rao"),
    ("Electronics and Communication", "ECE", "Dr. Vikram Menon"),
    ("Mechanical Engineering", "ME", "Dr. Neha Kulkarni"),
    ("Civil Engineering", "CE", "Dr. Arjun Iyer"),
    ("Business Administration", "MBA", "Dr. Meera Sharma"),
]

COURSE_TEMPLATES = {
    "CSE": [
        ("Programming Fundamentals", 4),
        ("Data Structures and Algorithms", 4),
        ("Database Management Systems", 3),
        ("Machine Learning", 3),
    ],
    "ECE": [
        ("Digital Logic Design", 4),
        ("Signals and Systems", 4),
        ("Communication Networks", 3),
        ("Embedded Systems", 3),
    ],
    "ME": [
        ("Engineering Mechanics", 4),
        ("Thermodynamics", 4),
        ("Manufacturing Processes", 3),
        ("Robotics and Automation", 3),
    ],
    "CE": [
        ("Engineering Surveying", 4),
        ("Structural Analysis", 4),
        ("Geotechnical Engineering", 3),
        ("Water Resources Engineering", 3),
    ],
    "MBA": [
        ("Principles of Management", 4),
        ("Managerial Economics", 4),
        ("Financial Accounting", 3),
        ("Business Analytics", 3),
    ],
}

FIRST_NAMES = [
    "Aarav", "Aditi", "Ananya", "Arjun", "Diya", "Ishaan", "Kavya", "Neha",
    "Rohan", "Saanvi", "Vihaan", "Meera", "Aditya", "Ira", "Kabir", "Nisha",
    "Rahul", "Sneha", "Varun", "Tara",
]
LAST_NAMES = [
    "Sharma", "Patel", "Rao", "Iyer", "Nair", "Gupta", "Reddy", "Mehta",
    "Kulkarni", "Menon", "Joshi", "Verma", "Das", "Kapoor", "Singh", "Khan",
]


def _grade_for_percentage(score_pct: float) -> str:
    if score_pct >= 90:
        return "O"
    if score_pct >= 80:
        return "A+"
    if score_pct >= 70:
        return "A"
    if score_pct >= 60:
        return "B+"
    if score_pct >= 50:
        return "B"
    if score_pct >= 40:
        return "C"
    if score_pct >= 35:
        return "P"
    return "F"


def seed_sample_data(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    seed: int = RANDOM_SEED,
) -> Path:
    """Initialize and fill the database with a repeatable 5/20/100 dataset.

    Exactly 25 students are assigned consistently low attendance, so 25% of
    the generated population appears in below-75% attendance queries.
    Re-running this function replaces the generated rows deterministically.
    """
    db_path = Path(db_path).expanduser().resolve()
    initialize_database(db_path)
    rng = random.Random(seed)

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            # Child tables are cleared first to satisfy foreign-key rules.
            connection.execute("DELETE FROM exam_results")
            connection.execute("DELETE FROM attendance")
            connection.execute("DELETE FROM courses")
            connection.execute("DELETE FROM students")
            connection.execute("DELETE FROM departments")

            connection.executemany(
                "INSERT INTO departments (dept_id, dept_name, dept_code, hod_name) "
                "VALUES (?, ?, ?, ?)",
                [(i, *row) for i, row in enumerate(DEPARTMENTS, start=1)],
            )

            course_rows: list[tuple[int, str, str, int, int, int]] = []
            course_id = 1
            for dept_id, (_, dept_code, _) in enumerate(DEPARTMENTS, start=1):
                for semester, (course_name, credits) in zip(
                    (1, 3, 5, 7), COURSE_TEMPLATES[dept_code]
                ):
                    course_rows.append(
                        (
                            course_id,
                            f"{dept_code}{semester}{course_id:02d}",
                            course_name,
                            dept_id,
                            credits,
                            semester,
                        )
                    )
                    course_id += 1
            connection.executemany(
                "INSERT INTO courses "
                "(course_id, course_code, course_name, dept_id, credits, semester) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                course_rows,
            )

            student_rows: list[tuple[int, str, str, str, int, int, float]] = []
            student_departments: dict[int, int] = {}
            student_semesters: dict[int, int] = {}
            low_attendance_students = set(rng.sample(range(1, 101), 25))
            used_names: set[str] = set()
            for student_id in range(1, 101):
                dept_id = ((student_id - 1) % len(DEPARTMENTS)) + 1
                dept_code = DEPARTMENTS[dept_id - 1][1]
                semester = rng.randint(1, 8)
                first = FIRST_NAMES[(student_id - 1) % len(FIRST_NAMES)]
                last = LAST_NAMES[((student_id - 1) // len(FIRST_NAMES)) % len(LAST_NAMES)]
                full_name = f"{first} {last}"
                # Make each display name unique while keeping it natural to read.
                if full_name in used_names:
                    full_name = f"{first} {last} {student_id:03d}"
                used_names.add(full_name)
                roll_no = f"{dept_code}{2022 + (student_id % 4)}{student_id:03d}"
                email = f"{dept_code.lower()}{student_id:03d}@campusmind.edu.in"
                cgpa = round(rng.uniform(5.2, 10.0), 2)
                student_rows.append(
                    (student_id, roll_no, full_name, email, dept_id, semester, cgpa)
                )
                student_departments[student_id] = dept_id
                student_semesters[student_id] = semester

            connection.executemany(
                "INSERT INTO students "
                "(student_id, roll_no, full_name, email, dept_id, semester, cgpa) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                student_rows,
            )

            attendance_rows: list[tuple[int, int, int, int, int, float]] = []
            exam_rows: list[tuple[int, int, int, str, float, float, str]] = []
            attendance_id = 1
            result_id = 1
            courses_by_dept: dict[int, list[tuple[int, int]]] = {}
            for course in course_rows:
                course_id, _, _, dept_id, _, course_semester = course
                courses_by_dept.setdefault(dept_id, []).append((course_id, course_semester))

            for student_id in range(1, 101):
                dept_id = student_departments[student_id]
                student_semester = student_semesters[student_id]
                # Each department has one course in terms 1, 3, 5, and 7.
                # Store current-term and prior-term outcomes for each student.
                for course_id, offered_semester in courses_by_dept[dept_id]:
                    if offered_semester > student_semester:
                        continue
                    classes_held = rng.randint(28, 42)
                    if student_id in low_attendance_students:
                        target_pct = rng.uniform(55.0, 72.5)
                    else:
                        target_pct = rng.uniform(76.0, 100.0)
                    classes_attended = min(
                        classes_held,
                        max(0, round(classes_held * target_pct / 100.0)),
                    )
                    attendance_pct = round(
                        100.0 * classes_attended / classes_held, 2
                    )
                    attendance_rows.append(
                        (
                            attendance_id,
                            student_id,
                            course_id,
                            classes_held,
                            classes_attended,
                            attendance_pct,
                        )
                    )
                    attendance_id += 1

                    for exam_type, max_marks in (("Midterm", 50.0), ("Final", 100.0)):
                        if student_id in low_attendance_students:
                            score_pct = rng.uniform(42.0, 83.0)
                        else:
                            score_pct = rng.uniform(56.0, 99.0)
                        marks = round(max_marks * score_pct / 100.0, 1)
                        exam_rows.append(
                            (
                                result_id,
                                student_id,
                                course_id,
                                exam_type,
                                marks,
                                max_marks,
                                _grade_for_percentage(100.0 * marks / max_marks),
                            )
                        )
                        result_id += 1

            connection.executemany(
                "INSERT INTO attendance "
                "(attendance_id, student_id, course_id, classes_held, "
                "classes_attended, attendance_pct) VALUES (?, ?, ?, ?, ?, ?)",
                attendance_rows,
            )
            connection.executemany(
                "INSERT INTO exam_results "
                "(result_id, student_id, course_id, exam_type, marks_obtained, "
                "max_marks, grade) VALUES (?, ?, ?, ?, ?, ?, ?)",
                exam_rows,
            )
    finally:
        connection.close()

    return db_path


if __name__ == "__main__":  # pragma: no cover - manual data-generation entry point
    output = seed_sample_data()
    print(f"Seeded CampusMind database: {output}")
