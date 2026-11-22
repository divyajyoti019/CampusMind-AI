"""Few-shot, schema-aware prompt construction for natural-language SQL."""

from __future__ import annotations


FEW_SHOT_EXAMPLES = """Example 1 - join and attendance threshold
Question: Show students whose attendance is below 75% in any course.
SQL:
SELECT s.roll_no, s.full_name, c.course_code, a.attendance_pct
FROM attendance AS a
JOIN students AS s ON s.student_id = a.student_id
JOIN courses AS c ON c.course_id = a.course_id
WHERE a.attendance_pct < 75
ORDER BY a.attendance_pct ASC
LIMIT 100;

Example 2 - aggregation by department
Question: What is the average CGPA for each department?
SQL:
SELECT d.dept_name, ROUND(AVG(s.cgpa), 2) AS average_cgpa
FROM students AS s
JOIN departments AS d ON d.dept_id = s.dept_id
GROUP BY d.dept_id, d.dept_name
ORDER BY average_cgpa DESC
LIMIT 100;

Example 3 - department filter and semester filter
Question: List the course names and credits offered by CSE in semester 3.
SQL:
SELECT c.course_code, c.course_name, c.credits
FROM courses AS c
JOIN departments AS d ON d.dept_id = c.dept_id
WHERE d.dept_code = 'CSE' AND c.semester = 3
ORDER BY c.course_code
LIMIT 100;

Example 4 - top performers across exam records
Question: Show the five students with the highest average exam percentage.
SQL:
SELECT s.roll_no, s.full_name,
       ROUND(AVG(100.0 * e.marks_obtained / e.max_marks), 2) AS average_exam_pct
FROM exam_results AS e
JOIN students AS s ON s.student_id = e.student_id
GROUP BY s.student_id, s.roll_no, s.full_name
ORDER BY average_exam_pct DESC
LIMIT 5;

Example 5 - grouped count with a join
Question: How many students are in each department?
SQL:
SELECT d.dept_name, COUNT(s.student_id) AS student_count
FROM departments AS d
LEFT JOIN students AS s ON s.dept_id = d.dept_id
GROUP BY d.dept_id, d.dept_name
ORDER BY student_count DESC
LIMIT 100;"""


def build_sql_prompt(
    question: str,
    schema_context: str,
    *,
    role: str = "Faculty / Administrator",
) -> str:
    """Inject inspected schema and five examples; constrain output to SQLite SQL."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not schema_context.strip():
        raise ValueError("schema_context must not be empty")

    if role.strip().lower() == "student":
        role_guidance = (
            "The requester is a student. Prefer aggregate summaries or a query "
            "scoped to the authenticated student's own records. Never expose "
            "another student's personal records, roll number, or email."
        )
    else:
        role_guidance = (
            "The requester is faculty or an administrator and may query academic "
            "records needed for university operations."
        )

    return f"""You are CampusMind AI's SQLite query writer.

Return exactly one executable SQLite SELECT statement and no markdown, prose,
comments, or explanations. The application will parse and validate your SQL.

Rules:
- Use only the tables and columns in the schema below.
- Use SQLite syntax and functions (for example, ROUND, AVG, COUNT, and LIMIT).
- Use explicit JOIN ... ON clauses for related tables.
- Do not use INSERT, UPDATE, DELETE, REPLACE, CREATE, DROP, ALTER, PRAGMA,
  ATTACH, DETACH, multiple statements, or any other non-SELECT operation.
- Do not reference tables outside the supplied schema.
- Do not use SELECT *; select only the columns needed to answer the question.
- Use a sensible ORDER BY for rankings and an explicit LIMIT for top-N requests.
- attendance.attendance_pct is already a percentage from 0 to 100; compare it
  directly with thresholds such as 75, without multiplying by 100 again.
- exam_results.marks_obtained is measured against exam_results.max_marks.
- If the question is ambiguous or asks for unavailable data, return a safe SELECT
  that returns no rows, such as SELECT NULL WHERE 1 = 0.

{role_guidance}

SQLite schema:
{schema_context}

Few-shot examples:
{FEW_SHOT_EXAMPLES}

User question:
{question.strip()}

SQL:"""
