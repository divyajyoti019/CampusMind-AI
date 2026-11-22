-- CampusMind AI normalized academic records schema (SQLite).
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS departments (
    dept_id      INTEGER PRIMARY KEY,
    dept_name    TEXT NOT NULL UNIQUE CHECK (length(trim(dept_name)) > 0),
    dept_code    TEXT NOT NULL UNIQUE CHECK (length(trim(dept_code)) > 0),
    hod_name     TEXT NOT NULL CHECK (length(trim(hod_name)) > 0)
);

CREATE TABLE IF NOT EXISTS students (
    student_id   INTEGER PRIMARY KEY,
    roll_no      TEXT NOT NULL UNIQUE CHECK (length(trim(roll_no)) > 0),
    full_name    TEXT NOT NULL CHECK (length(trim(full_name)) > 0),
    email        TEXT NOT NULL UNIQUE CHECK (email LIKE '%_@_%._%'),
    dept_id      INTEGER NOT NULL,
    semester     INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    cgpa         REAL NOT NULL CHECK (cgpa BETWEEN 0.0 AND 10.0),
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS courses (
    course_id    INTEGER PRIMARY KEY,
    course_code  TEXT NOT NULL UNIQUE CHECK (length(trim(course_code)) > 0),
    course_name  TEXT NOT NULL CHECK (length(trim(course_name)) > 0),
    dept_id      INTEGER NOT NULL,
    credits      INTEGER NOT NULL CHECK (credits BETWEEN 1 AND 6),
    semester     INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS attendance (
    attendance_id    INTEGER PRIMARY KEY,
    student_id       INTEGER NOT NULL,
    course_id        INTEGER NOT NULL,
    classes_held     INTEGER NOT NULL CHECK (classes_held > 0),
    classes_attended INTEGER NOT NULL CHECK (
        classes_attended >= 0 AND classes_attended <= classes_held
    ),
    attendance_pct   REAL NOT NULL CHECK (attendance_pct BETWEEN 0.0 AND 100.0),
    CHECK (
        ABS(attendance_pct - ROUND(100.0 * classes_attended / classes_held, 2)) <= 0.01
    ),
    UNIQUE (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS exam_results (
    result_id      INTEGER PRIMARY KEY,
    student_id     INTEGER NOT NULL,
    course_id      INTEGER NOT NULL,
    exam_type      TEXT NOT NULL CHECK (exam_type IN ('Midterm', 'Final')),
    marks_obtained REAL NOT NULL CHECK (marks_obtained >= 0.0),
    max_marks      REAL NOT NULL CHECK (max_marks > 0.0),
    grade          TEXT NOT NULL CHECK (grade IN ('O', 'A+', 'A', 'B+', 'B', 'C', 'P', 'F')),
    CHECK (marks_obtained <= max_marks),
    UNIQUE (student_id, course_id, exam_type),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_students_department_semester
    ON students(dept_id, semester);
CREATE INDEX IF NOT EXISTS idx_courses_department_semester
    ON courses(dept_id, semester);
CREATE INDEX IF NOT EXISTS idx_attendance_student
    ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_course_pct
    ON attendance(course_id, attendance_pct);
CREATE INDEX IF NOT EXISTS idx_exam_results_student
    ON exam_results(student_id);
CREATE INDEX IF NOT EXISTS idx_exam_results_course_type
    ON exam_results(course_id, exam_type);
