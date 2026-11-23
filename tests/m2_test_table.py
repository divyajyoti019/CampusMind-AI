"""Milestone 2 test table: run each test input against the real code (offline mode) and save actual outputs.

Run from the repository root:  python tests/m2_test_table.py
Writes docs/milestone2/test_results.json and prints one line per test.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import settings
from modules.intent_router import IntentRouter
from utils.security import validate_and_sanitize_sql, enforce_student_query_scope
from database.db import execute_readonly_query
from utils.pdf_processor import extract_pdf_pages
from modules.rag_assistant import RAGAssistant
from modules.analytics import compute_kpis, compute_summary_statistics, _contains_only_supported_numbers

rows = []
def add(tid, kind, module, inp, expected, actual, ok):
    rows.append(dict(id=tid, type=kind, module=module, input=inp, expected=expected, actual=actual, result="Pass" if ok else "Fail"))

router = IntentRouter(use_llm=False)
def route(q, role="Faculty / Administrator"):
    p = router.route(q, role=role); return p
for tid, kind, q, role, exp in [
    ("T01","Normal","What is the university minimum attendance requirement?","Faculty / Administrator",["rag"]),
    ("T02","Normal","Show students with attendance below 75%.","Faculty / Administrator",["sql"]),
    ("T03","Normal","Show students below 75% attendance and explain the university attendance policy.","Faculty / Administrator",["rag","sql","analytics"]),
    ("T04","Normal","What are the tuition payment and late-fee rules?","Faculty / Administrator",["rag"]),
]:
    p = route(q, role); add(tid, kind, "Intent router", f"[{role}] {q}", "Intents: " + " + ".join(exp), "Intents: " + " + ".join(p.intents), sorted(p.intents)==sorted(exp))
p = route("Show students with attendance below 75%.", "Student")
add("T05","Edge","Intent router (role check)","[Student] Show students with attendance below 75%.","SQL blocked; access message shown",
    f"Intents: {' + '.join(p.intents) or 'none'}; blocked: {', '.join(p.blocked_intents)}; message shown: {'yes' if p.access_message else 'no'}",
    "sql" in (p.blocked_intents or []) and bool(p.access_message))
p = route("What is the average CGPA by department?", "Student")
add("T06","Edge","Intent router (role check)","[Student] What is the average CGPA by department?","SQL allowed (aggregate scope)",
    f"Intents: {' + '.join(p.intents)}; scope: {p.sql_scope}; blocked: {', '.join(p.blocked_intents) or 'none'}", "sql" in p.intents and p.sql_scope=="aggregate")

def run_sql(sql, role="Faculty / Administrator"):
    try:
        safe = validate_and_sanitize_sql(sql)
        if role == "Student":
            safe = enforce_student_query_scope(safe)
        df = execute_readonly_query(safe, db_path=settings.database_path)
        return True, df, safe
    except Exception as e:
        return False, str(e), None
ok, df, _ = run_sql("SELECT d.dept_name, COUNT(s.student_id) AS student_count FROM departments d LEFT JOIN students s ON s.dept_id = d.dept_id GROUP BY d.dept_id, d.dept_name")
add("T07","Normal","SQL guardrail + SQLite","SELECT dept_name, COUNT(student_id) ... GROUP BY department","Accepted; 5 rows (20 students each)",
    f"Accepted; {len(df)} rows; counts = {sorted(set(df['student_count'].tolist()))}" if ok else df, ok and len(df)==5)
ok, df, _ = run_sql("SELECT COUNT(DISTINCT student_id) AS low_attendance_students FROM attendance WHERE attendance_pct < 75")
add("T08","Normal","SQL guardrail + SQLite","SELECT COUNT(DISTINCT student_id) FROM attendance WHERE attendance_pct < 75","Accepted; 25",
    f"Accepted; result = {int(df.iloc[0,0])}" if ok else df, ok and int(df.iloc[0,0])==25)
for tid, sql in [("T09","DROP TABLE students"),("T10","SELECT student_id FROM students; DROP TABLE students"),("T11","SELECT name FROM sqlite_master WHERE type = 'table'")]:
    ok, err, _ = run_sql(sql)
    add(tid,"Invalid","SQL guardrail",sql,"Rejected before execution", "Accepted (unsafe!)" if ok else f"Rejected: {err}", not ok)
ok, df, safe = run_sql("SELECT roll_no, full_name FROM students LIMIT 5000")
add("T12","Edge","SQL guardrail (row cap)","SELECT roll_no, full_name FROM students LIMIT 5000","LIMIT capped at 100",
    f"Rewritten to '...LIMIT {safe.rsplit('LIMIT',1)[1].strip()}'; {len(df)} rows returned" if ok else df, ok and len(df)==100)
ok, err, _ = run_sql("SELECT GROUP_CONCAT(full_name || ' ' || cgpa, '; ') AS x FROM students", role="Student")
add("T13","Invalid","Student privacy check","[Student] SELECT GROUP_CONCAT(full_name || ' ' || cgpa) AS x FROM students","Rejected (would list names + CGPA)",
    "Accepted (leak!)" if ok else f"Rejected: {err}", not ok)
ok, err, _ = run_sql("SELECT MAX(CASE WHEN full_name = 'Aarav Sharma' THEN cgpa END) AS cgpa FROM students", role="Student")
add("T14","Invalid","Student privacy check","[Student] SELECT MAX(CASE WHEN full_name='Aarav Sharma' THEN cgpa END) FROM students","Rejected (singles out one student)",
    "Accepted (leak!)" if ok else f"Rejected: {err}", not ok)
ok, df, _ = run_sql("SELECT d.dept_name, ROUND(AVG(s.cgpa), 2) AS average_cgpa FROM students s JOIN departments d ON d.dept_id = s.dept_id GROUP BY d.dept_id, d.dept_name", role="Student")
add("T15","Edge","Student privacy check","[Student] SELECT dept_name, ROUND(AVG(cgpa),2) ... GROUP BY department","Accepted (aggregate only)",
    f"Accepted; {len(df)} rows, e.g. {df.iloc[0,0]} = {df.iloc[0,1]}" if ok else f"Rejected: {df}", ok and len(df)==5)

pages = extract_pdf_pages(settings.documents_dir / "University_Academic_Handbook.pdf")
def page_has(n, terms):
    t = next(p for p in pages if p["page"]==n)["text"].casefold(); return [x for x in terms if x.casefold() in t]
for tid, q, n, terms in [("T16","How does medical condonation work for attendance?",1,["65%","seven calendar days","Dean (Academic Affairs)"]),
                          ("T17","What are the tuition payment and late-fee rules?",3,["10th calendar day","INR 100","INR 1,500"])]:
    found = page_has(n, terms)
    add(tid,"Normal","Policy source check (handbook)",q,f"Page {n} contains: {', '.join(terms)}",f"Found {len(found)}/{len(terms)} facts on page {n}",len(found)==len(terms))
chunks = [{"source":"University_Academic_Handbook.pdf","page":1,"text":"..."}]
ans_bad = "Minimum attendance is 75% [University_Academic_Handbook.pdf, Page 9]."
g = RAGAssistant._citations_are_grounded(ans_bad, chunks)
add("T18","Edge","RAG citation guardrail","Model reply citing Page 9 when only Page 1 was retrieved","Reply rejected -> fixed refusal message",
    "Rejected -> refusal shown" if not g else "Accepted (hallucinated citation!)", not g)
ans_none = "You need 75% attendance in each course."
g = RAGAssistant._citations_are_grounded(ans_none, chunks)
add("T19","Invalid","RAG citation guardrail","Model reply with no citation","Reply rejected -> fixed refusal message",
    "Rejected -> refusal shown" if not g else "Accepted", not g)
ans_ok = "You must attend at least 75% of classes in each registered course [University_Academic_Handbook.pdf, Page 1]."
g = RAGAssistant._citations_are_grounded(ans_ok, chunks)
add("T20","Normal","RAG citation guardrail","Model reply citing the retrieved Page 1","Reply accepted",
    "Accepted" if g else "Rejected", g)

k = compute_kpis(settings.database_path)
add("T21","Normal","Analytics KPIs","Compute dashboard KPIs from the sample database","100 students; 25 below 75%",
    f"{k['total_students']} students; avg attendance {k['overall_avg_attendance_pct']}%; avg score {k['overall_avg_score_pct']}%; {k['students_below_attendance_threshold']} below 75%",
    k['total_students']==100 and k['students_below_attendance_threshold']==25)
stats = compute_summary_statistics(settings.database_path)
bad = f"Average attendance is {k['overall_avg_attendance_pct']}% and rose 12% since last year."
ok_bad = _contains_only_supported_numbers(bad, stats)
add("T22","Edge","Analytics number guardrail",f"Summary text: \"{bad}\"","Rejected (12 is not in the computed statistics)",
    "Accepted (invented number!)" if ok_bad else "Rejected: unsupported numeric value", not ok_bad)
good = f"{k['students_below_attendance_threshold']} students are below the threshold and average attendance is {k['overall_avg_attendance_pct']}%."
ok_good = _contains_only_supported_numbers(good, stats)
add("T23","Normal","Analytics number guardrail",f"Summary text: \"{good}\"","Accepted (all numbers come from the data)",
    "Accepted" if ok_good else "Rejected", ok_good)

for tid, q, exp in [("T24","Show me the exam rules for arriving late.",["rag"]),("T25","Which department has the lowest attendance?",["sql","analytics"])]:
    p = route(q); add(tid,"Edge","Intent router (paraphrase)",f"[Faculty / Administrator] {q}","Intents: " + " + ".join(exp),"Intents: " + " + ".join(p.intents),sorted(p.intents)==sorted(exp))

(ROOT / "docs" / "milestone2").mkdir(parents=True, exist_ok=True)
json.dump(dict(rows=rows, kpis=k, correlation=stats["attendance_exam_score_correlation"], dept=stats["department_attendance"]), open(ROOT / "docs" / "milestone2" / "test_results.json", "w"), indent=1)
for r in rows: print(r["id"], r["type"], r["result"], "|", r["actual"][:110])
print(sum(r["result"]=="Pass" for r in rows), "/", len(rows))
