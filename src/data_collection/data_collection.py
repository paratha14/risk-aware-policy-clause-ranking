import os
import re
import requests
from tqdm import tqdm
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw_docs")
PRIVACY_DIR  = os.path.join(RAW_DIR, "privacy")
ACADEMIC_DIR = os.path.join(RAW_DIR, "academic")
HR_DIR       = os.path.join(RAW_DIR, "hr")

for d in [PRIVACY_DIR, ACADEMIC_DIR, HR_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("STEP 1 — DOCUMENT COLLECTION")
print("=" * 60)


# ============================================================
# PRIVACY POLICIES
# ============================================================
print("\n[1/3] Collecting privacy policy documents...")

# Try HuggingFace first
try:
    priv = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    saved = 0
    for i, item in enumerate(tqdm(priv, desc="  Saving")):
        text = str(item.get("text", "")).strip()
        if len(text.split()) < 30:
            continue
        with open(os.path.join(PRIVACY_DIR, f"ledgar_{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        saved += 1
        if saved >= 40:
            break
    print(f"  ✓ Saved {saved} texts from lex_glue")
except Exception as e:
    print(f"  ✗ lex_glue failed: {str(e)[:60]}")

# Curated samples — always saved
PRIVACY_DOCS = {
    "gdpr_policy": """GDPR DATA PROTECTION POLICY

Article 1 — Retention Periods
Personal data shall not be kept longer than necessary for its stated purpose.
Violation of retention limits constitutes a breach under GDPR Article 5(1)(e)
and may result in fines up to €20 million or 4% of annual global turnover.

Article 2 — Right to Erasure
Data subjects may request erasure of personal data without undue delay.
Controllers must comply within 30 days. Non-compliance is subject to regulatory
enforcement and administrative fines.

Article 3 — Data Breach Notification
Personal data breaches must be reported to supervisory authorities within 72
hours. Failure to notify constitutes an infringement attracting administrative
fines and potential criminal liability.

Article 4 — Cross-Border Transfers
Transfer of personal data to third countries lacking adequate protections is
prohibited. Unauthorized transfers may result in suspension of processing
activities and significant financial penalties.""",

    "ccpa_policy": """CALIFORNIA CONSUMER PRIVACY ACT — COMPLIANCE POLICY

Section 1 — Consumer Rights
California residents have the right to know what personal information is collected.
Businesses must respond to verifiable requests within 45 days. Non-compliance
may result in civil penalties of $2,500 per unintentional violation and $7,500
per intentional violation enforced by the Attorney General.

Section 2 — Opt-Out Rights
Consumers may opt out of the sale of personal information at any time. Businesses
must honor opt-out requests immediately and may not deny services to consumers
exercising this right.

Section 3 — Data Security
Businesses must implement reasonable security procedures. Failure to maintain
adequate security is subject to civil litigation and statutory damages of $100
to $750 per consumer per incident.

Section 4 — Minors
Collection of data from children under 16 requires explicit opt-in consent.
Violations involving minors data attract enhanced penalties.""",

    "cookie_policy": """COOKIE AND TRACKING POLICY

1. Consent Requirement
Analytics and marketing cookies require explicit user consent before activation.
Deploying non-essential cookies without consent violates ePrivacy regulations
and may result in regulatory fines.

2. Third-Party Sharing
Data collected via cookies shall not be shared with third parties without
user consent. Unauthorized sharing constitutes a breach of applicable data
protection laws and may result in enforcement action.

3. Retention Limits
Cookies must not be retained beyond their stated purpose or declared lifespan.
Exceeding declared retention periods constitutes a policy violation subject to
regulatory audit.

4. Children's Protection
Tracking cookies must not be deployed on pages directed at children under 13.
Violations will result in mandatory reporting and enhanced regulatory scrutiny.""",

    "employee_data_policy": """EMPLOYEE DATA PRIVACY POLICY

1. Data Collection Scope
Employee personal data is collected only for lawful employment purposes.
Collection beyond stated purposes is prohibited and constitutes a violation.

2. Monitoring Policy
Electronic monitoring of employee communications requires prior notification.
Covert monitoring without legal basis may expose the organisation to liability.

3. Access Controls
Employee data is accessible only to authorised HR personnel and direct managers.
Unauthorised access constitutes a serious disciplinary offence and may result
in termination.

4. Retention After Termination
Employee data shall be retained for no longer than 7 years after termination
unless required by law. Data retained beyond this period must be deleted upon
request.""",
}

for name, text in PRIVACY_DOCS.items():
    with open(os.path.join(PRIVACY_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
print(f"  ✓ Saved {len(PRIVACY_DOCS)} curated privacy docs")


# ============================================================
# ACADEMIC / UNIVERSITY POLICIES
# ============================================================
print("\n[2/3] Collecting academic policy documents...")

ACADEMIC_HTML = [
    ("stanford_honor_code",  "https://communitystandards.stanford.edu/policies-guidance/honor-code"),
    ("mit_academic_conduct", "https://integrity.mit.edu/handbook/academic-misconduct"),
    ("harvard_integrity",    "https://college.harvard.edu/academics/academic-integrity"),
    ("oxford_plagiarism",    "https://www.ox.ac.uk/students/academic/guidance/skills/plagiarism"),
]

headers   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
html_saved = 0

for name, url in tqdm(ACADEMIC_HTML, desc="  Scraping"):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            text = re.sub(r'<[^>]+>', ' ', r.text)
            text = re.sub(r'&[a-z]+;', ' ', text)
            text = re.sub(r'\s{3,}', '\n\n', text).strip()
            if len(text.split()) > 150:
                with open(os.path.join(ACADEMIC_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
                html_saved += 1
                print(f"    ✓ {name}")
            else:
                print(f"    ✗ {name} — too little text")
        else:
            print(f"    ✗ {name} — HTTP {r.status_code}")
    except Exception as e:
        print(f"    ✗ {name} — {str(e)[:50]}")

ACADEMIC_DOCS = {
    "exam_and_assessment_policy": """EXAMINATION AND ASSESSMENT POLICY

1. Academic Integrity
Students must not engage in any form of academic dishonesty including plagiarism,
fabrication, cheating, or unauthorized collaboration. Violations may result in
course failure, suspension, or permanent expulsion. All violations are recorded
on the student's academic file.

2. Examination Conduct
Mobile phones and unauthorized electronic devices are strictly prohibited during
examinations. Students found with unauthorized materials shall receive a zero for
the entire examination and be referred to the Academic Integrity Committee.

3. Plagiarism
Presenting another person's work as one's own constitutes plagiarism. Confirmed
plagiarism results in a failing grade. Repeat offences result in academic
suspension for a minimum of one semester.

4. Use of AI Tools
Submission of AI-generated text without disclosure is considered academic
dishonesty. First offence results in assignment failure. Repeat offences result
in course failure and referral to the Disciplinary Committee.

5. Late Submission
Assignments submitted after the deadline without an approved extension receive
a deduction of 10% per calendar day. No submission accepted after 7 days without
documented medical or compassionate grounds.

6. Attendance
Students must attend at least 75% of scheduled classes. Students below this
threshold are barred from final examinations without appeal.""",

    "student_conduct_code": """STUDENT CODE OF CONDUCT

Section 1 — Prohibited Conduct
The following conduct is strictly prohibited and will result in disciplinary action:
(a) Physical violence or threats of violence against any university community member
(b) Harassment, bullying, or discriminatory behaviour on any protected ground
(c) Unauthorized access to university IT systems or restricted facilities
(d) Theft or wilful damage to university or personal property
(e) Possession, use, or distribution of illegal substances on campus
(f) Forgery or misrepresentation in academic or administrative matters

Section 2 — Sanctions
Students found in violation may face:
- Formal written warning on academic record
- Suspension of campus privileges
- Academic suspension for up to two semesters
- Permanent expulsion from the university

Section 3 — Appeals
Students may appeal decisions within 14 working days of notification. Appeals
must be submitted in writing with supporting documentation. The Appeals
Committee decision is final and binding.

Section 4 — Substance Misuse
Possession or distribution of illegal substances is a critical violation.
First-time offenders face mandatory counselling and probation. Repeat offenders
are subject to immediate suspension and possible expulsion.""",

    "course_syllabus_policies": """COURSE POLICIES AND EXPECTATIONS

Grading
  Midterm Examination:  25%
  Final Examination:    35%
  Assignments (x4):     25%
  Participation:        15%

Minimum passing grade is 50%. Students who fail must retake the course before
proceeding to dependent modules.

Academic Honesty
All submitted work must be the student's own. Unauthorized use of AI tools,
copying, or submitting previously graded work violates the Academic Integrity
Policy. First violation results in a zero. Second violation results in course
failure.

Submissions
Assignments must be submitted via the course portal before 11:59 PM on the
due date. Late submissions without approved extensions will not be accepted
after 7 days.

Withdrawal
Students wishing to withdraw must do so before the published deadline.
Unofficial withdrawal results in an automatic F on the permanent transcript.""",

    "library_and_it_policy": """LIBRARY AND IT ACCEPTABLE USE POLICY

1. IT Systems Access
University IT systems are for academic and administrative use only. Attempts to
compromise system security are serious violations subject to disciplinary action
and potential criminal prosecution.

2. Software and Copyright
Installation of unlicensed software on university equipment is prohibited.
Distributing copyrighted materials without authorization subjects the student
to legal liability.

3. Library Resources
Library materials must be returned by the due date. Theft or deliberate damage
is a disciplinary offence that may result in suspension of library privileges
and financial restitution.

4. Data and Privacy
Students must not access or modify other users data without authorization.
Unauthorized access constitutes a serious breach of policy subject to immediate
disciplinary action.""",
}

for name, text in ACADEMIC_DOCS.items():
    with open(os.path.join(ACADEMIC_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
print(f"  ✓ Saved {html_saved} scraped + {len(ACADEMIC_DOCS)} curated academic docs")


# ============================================================
# HR / CORPORATE POLICIES
# ============================================================
print("\n[3/3] Saving HR/corporate policy documents...")

HR_DOCS = {
    "hr_code_of_conduct": """EMPLOYEE CODE OF CONDUCT

1. Attendance and Punctuality
Employees must report to work on time and maintain regular attendance. Three
unexcused absences within 30 days constitute grounds for immediate dismissal.

2. Confidentiality
Employees must not disclose confidential company information during or after
employment. Violations result in immediate termination and may lead to civil
litigation and claims for damages.

3. Workplace Harassment
Harassment of any kind is strictly prohibited. Employees found guilty face
suspension without pay or termination. Serious cases are referred directly
to law enforcement.

4. Company Resources
Company equipment must be used for business purposes only. Misuse or theft
results in immediate termination and potential criminal prosecution.

5. Conflict of Interest
Employees must disclose conflicts of interest immediately. Undisclosed conflicts
including financial relationships with competitors result in termination for
cause and possible civil action.

6. Social Media
Employees must not post confidential or defamatory content online. Violations
may result in disciplinary action, termination, and legal proceedings.""",

    "hr_disciplinary_policy": """DISCIPLINARY POLICY

1. Progressive Discipline
  Step 1: Verbal warning (documented in writing)
  Step 2: Written warning
  Step 3: Final written warning or suspension without pay (up to 5 days)
  Step 4: Termination of employment

2. Gross Misconduct
Gross misconduct results in immediate dismissal without notice. Examples include
assault, intoxication at work, fraud, destruction of property, or serious
confidentiality breaches.

3. Performance Management
Employees failing to meet targets after a 60-day Performance Improvement Plan
may be dismissed for incapability.

4. Appeal Rights
Employees may appeal any disciplinary decision within 10 working days of written
notification. The HR Director decision on appeal is final.""",

    "hr_data_and_it_policy": """EMPLOYEE IT AND DATA POLICY

1. Acceptable Use
Company IT systems are for business use only. Accessing inappropriate or illegal
content on company systems is a gross misconduct offence.

2. Password Security
Employees must not share login credentials. Sharing passwords resulting in a
security breach is treated as gross misconduct subject to immediate dismissal.

3. Data Handling
Unauthorized disclosure or loss of personal data is a serious disciplinary
offence and may constitute a criminal offence.

4. Monitoring
The company reserves the right to monitor use of company IT systems. Employees
have no expectation of privacy on company equipment or networks.

5. Software
Installation of unauthorized software is prohibited. Violations may result in
disciplinary action and remediation costs charged to the employee.""",

    "hr_leave_policy": """LEAVE OF ABSENCE POLICY

1. Annual Leave
Employees are entitled to 20 days paid annual leave per year. Carry-forward of
more than 5 days requires written management approval.

2. Sick Leave
Employees receive 10 days paid sick leave per year. Medical certification is
required for absences exceeding 3 consecutive days. Misuse constitutes
misconduct subject to disciplinary action including dismissal.

3. Unauthorized Absence
Absence for 3 or more consecutive days without notification constitutes
abandonment of employment and may result in automatic termination.

4. Leave During Notice Period
Annual leave cannot be taken during the notice period without explicit written
HR approval. Unauthorized leave during notice is deducted from final salary.""",
}

for name, text in HR_DOCS.items():
    with open(os.path.join(HR_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
print(f"  ✓ Saved {len(HR_DOCS)} HR policy docs")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("COLLECTION SUMMARY")
print("=" * 60)

total = 0
for domain, folder in [("Legal (manual)",  os.path.join(RAW_DIR, "legal")),
                        ("Privacy",         PRIVACY_DIR),
                        ("Academic",        ACADEMIC_DIR),
                        ("HR / Corporate",  HR_DIR)]:
    folder_exists = os.path.isdir(folder)
    count = len([f for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f))]) if folder_exists else 0
    total += count
    print(f"  {'✓' if count > 0 else '✗'} {domain:<22} {count:>4} files")

print(f"\n  {'TOTAL':<25} {total:>4} files")