import os
import re
import pandas as pd
from tqdm import tqdm

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    print("⚠  spaCy not available — long paragraphs won't be split")
    print("   Run: pip install spacy && python -m spacy download en_core_web_sm")

LONG_PARA_WORDS = 40  # paragraphs over this get spaCy sentence splitting

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠  beautifulsoup4 not found — HTML docs will use regex fallback")
    print("   Run: pip install beautifulsoup4 lxml")

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXTRACTED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "segmented")
os.makedirs(SEGMENTED_DIR, exist_ok=True)

MIN_WORDS = 10
MAX_WORDS = 120
MAX_CLAUSES_PER_DOC = 120  

print("=" * 60)
print("HYBRID CLAUSE SEGMENTATION (Rule-based + spaCy)")
print("=" * 60)


# ============================================================
# HELPERS
# ============================================================

def is_html(text):
    return bool(re.search(r'<(html|body|div|p|head|span|table)[^>]*>', text, re.IGNORECASE))

def strip_html(text):
    if BS4_AVAILABLE:
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    else:
        text = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        return text

def clean_clause(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' \t\n\r.,;:')

def is_valid(text):
    wc = len(text.split())
    return MIN_WORDS <= wc <= MAX_WORDS

def spacy_split(text):
    """Split long paragraph into sentences using spaCy."""
    if not SPACY_AVAILABLE:
        return [text]
    doc = nlp(text[:900000])
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# ============================================================
# PATTERNS
# ============================================================

# Shared — catches BOTH line-start and inline numbered items
# e.g.  "1. text"  or  text running into "2. next clause"
NUMBERED       = re.compile(r'(?m)(?:^|\s)(\d+(\.\d+)*)\.\s+(?=\S)')
SUBBULLET      = re.compile(r'(?m)^\s*(-{1,2}|[*•→▪‣–])\s+(?=\S)')

# Legal — lettered parens only after newline or sentence boundary
# Avoids matching mid-sentence e.g. "company (a division of...)"
LETTERED_PAREN = re.compile(r'(?:^|\n)\s*\([a-zA-Z]\)\s+(?=\S)', re.IGNORECASE)
EXHIBIT        = re.compile(r'(?m)^\s*[Ee][Xx][Hh][Ii][Bb][Ii][Tt]\s+[\dA-Z][\d\.]*\s*$')
DEFINITION     = re.compile(r'(?m)^\s*["\u201c][A-Z][^""\u201d]{1,60}["\u201d]\s+(means|shall mean|refers to|is defined as)')
INITIALS_LINE  = re.compile(r'(?i)initials?\s*[_\-]{2,}')

# Privacy — also catches inline Section/Article references
SECTION_ARTICLE = re.compile(
    r'(?:^|\n)\s*(Section|Article|Amendment|Clause|Schedule)\s+[\dIVXivx]+[\d\.]*',
    re.IGNORECASE
)
LEGAL_KEYWORDS = re.compile(
    r'\b(dispute|claim|controversy|violation|investigation|notice|request|demand|'
    r'obligation|indemnif|terminat|arbitrat|liabilit|penalt|sancti|enforce|'
    r'prohibit|restrict|disclos)\w*\b',
    re.IGNORECASE
)


# ============================================================
# SEGMENTERS
# ============================================================

def segment_academic_hr(text):
    results = []

    if is_html(text):
        text = strip_html(text)

    paragraphs = re.split(r'\n{2,}', text)

    for para in paragraphs:
        para = para.strip()
        if not para or len(para.split()) <= 4:
            continue

        num_match = re.match(r'^\s*(\d+(\.\d+)*)\.\s+', para)

        if num_match:
            lines        = para.splitlines()
            heading_text = re.sub(r'^\s*\d+(\.\d+)*\.\s+', '', lines[0]).strip()
            sub_clauses  = []

            for line in lines[1:]:
                line = line.strip()
                if SUBBULLET.match(line):
                    bullet_text = re.sub(r'^[-•*→]\s+', '', line).strip()
                    combined    = f"{heading_text}: {bullet_text}" if heading_text else bullet_text
                    sub_clauses.append((combined, "semi_clause_bullet"))
                elif line:
                    heading_text = f"{heading_text} {line}".strip()

            if sub_clauses:
                results.extend(sub_clauses)
            else:
                results.append((heading_text, "numbered_clause"))

        else:
            bullets = SUBBULLET.split(para)
            if len(bullets) > 1:
                for b in bullets:
                    b = b.strip()
                    if b:
                        results.append((b, "bullet_clause"))
            else:
                # spaCy fallback for long unstructured paragraphs
                if SPACY_AVAILABLE and len(para.split()) > LONG_PARA_WORDS:
                    for sent in spacy_split(para):
                        results.append((sent, "spacy_sentence"))
                else:
                    results.append((para, "paragraph"))

    return results


def segment_legal(text):
    results = []

    if is_html(text):
        text = strip_html(text)

    # Remove noise lines
    text = EXHIBIT.sub('', text)
    text = INITIALS_LINE.sub('', text)

    paragraphs = re.split(r'\n{2,}', text)

    for para in paragraphs:
        para = para.strip()
        if not para or len(para.split()) <= 3:
            continue

        # Definitions first
        if DEFINITION.search(para):
            results.append((para, "definition_clause"))
            continue

        # Lettered sub-clauses (a) (b)
        lettered_parts = LETTERED_PAREN.split(para)
        if len(lettered_parts) > 1:
            for part in lettered_parts:
                part = part.strip()
                if part:
                    results.append((part, "lettered_subclause"))
            continue

        # Numbered clauses
        numbered_parts = NUMBERED.split(para)
        if len(numbered_parts) > 1:
            clean_parts = [p.strip() for p in numbered_parts
                           if p and not re.fullmatch(r'\d+(\.\d+)*', p.strip())]
            for part in clean_parts:
                if part:
                    results.append((part, "numbered_clause"))
            continue

        # spaCy fallback for long legal paragraphs
        if SPACY_AVAILABLE and len(para.split()) > LONG_PARA_WORDS:
            for sent in spacy_split(para):
                results.append((sent, "spacy_sentence"))
        else:
            results.append((para, "paragraph"))

    return results


def segment_privacy(text):
    if is_html(text):
        text = strip_html(text)

    # Try Section/Article split first (old-style privacy docs)
    parts = [p.strip() for p in SECTION_ARTICLE.split(text) if p.strip()]

    # If no Section/Article structure — doc uses numbered clauses
    # (HIPAA, GDPR, COPPA, BIPA etc.) — route through academic_hr
    if len(parts) <= 1:
        raw = segment_academic_hr(text)
        return [
            (c, "legal_keyword_clause" if LEGAL_KEYWORDS.search(c) else lbl)
            for c, lbl in raw
        ]

    results = []
    for part in parts:
        if not part or len(part.split()) <= 4:
            continue

        numbered_parts = NUMBERED.split(part)
        if len(numbered_parts) > 1:
            clean_parts = [p.strip() for p in numbered_parts
                           if p and not re.fullmatch(r'\d+(\.\d+)*', p.strip())]
            for np in clean_parts:
                if np:
                    label = "legal_keyword_clause" if LEGAL_KEYWORDS.search(np) else "numbered_clause"
                    results.append((np, label))
        else:
            label = "legal_keyword_clause" if LEGAL_KEYWORDS.search(part) else "section_clause"
            if SPACY_AVAILABLE and len(part.split()) > LONG_PARA_WORDS:
                for sent in spacy_split(part):
                    results.append((sent, label))
            else:
                results.append((part, label))

    return results


SEGMENTERS = {
    "academic" : segment_academic_hr,
    "hr"       : segment_academic_hr,
    "legal"    : segment_legal,
    "privacy"  : segment_privacy,
}

def segment(text, domain):
    return SEGMENTERS.get(domain, segment_academic_hr)(text)


# ============================================================
# MAIN
# ============================================================
all_clauses  = []
domain_stats = {}

domains = sorted([
    d for d in os.listdir(EXTRACTED_DIR)
    if os.path.isdir(os.path.join(EXTRACTED_DIR, d))
])

for domain in domains:
    domain_dir = os.path.join(EXTRACTED_DIR, domain)
    files      = [f for f in os.listdir(domain_dir) if f.endswith(".txt")]

    print(f"\n[{domain.upper()}] — {len(files)} documents")

    domain_count = 0
    skipped      = 0

    for fname in tqdm(files, desc="  Segmenting"):
        fpath = os.path.join(domain_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        if len(text.strip()) < 50:
            skipped += 1
            continue

        doc_clauses = 0
        for clause_text, pattern_label in segment(text, domain):
            if doc_clauses >= MAX_CLAUSES_PER_DOC:
                break
            clause_text = clean_clause(clause_text)
            if not is_valid(clause_text):
                continue
            all_clauses.append({
                "clause_id"     : f"C{len(all_clauses) + 1:06d}",
                "source_doc"    : os.path.splitext(fname)[0],
                "domain"        : domain,
                "raw_text"      : clause_text,
                "word_count"    : len(clause_text.split()),
                "pattern_label" : pattern_label,
            })
            domain_count += 1
            doc_clauses  += 1

    domain_stats[domain] = domain_count
    print(f"  ✓ {domain_count} clauses  |  {skipped} docs skipped")


# ============================================================
# PASS 2 — spaCy sentence splitting on surviving paragraphs
# Any clause tagged "paragraph" that is still over MAX_WORDS
# gets re-split into individual sentences here
# ============================================================
print("\nPass 2 — spaCy re-split on long paragraphs...")

if SPACY_AVAILABLE:
    refined = []
    resplit  = 0

    for clause in all_clauses:
        if (clause["pattern_label"] == "paragraph"
                and clause["word_count"] > LONG_PARA_WORDS):
            sents = spacy_split(clause["raw_text"])
            if len(sents) > 1:
                resplit += 1
                for sent in sents:
                    sent = clean_clause(sent)
                    if is_valid(sent):
                        new_clause = clause.copy()
                        new_clause["raw_text"]      = sent
                        new_clause["word_count"]    = len(sent.split())
                        new_clause["pattern_label"] = "spacy_pass2"
                        refined.append(new_clause)
            else:
                refined.append(clause)
        else:
            refined.append(clause)

    # Re-assign sequential clause IDs
    for i, clause in enumerate(refined):
        clause["clause_id"] = f"C{i + 1:06d}"

    all_clauses = refined
    print(f"  ✓ Re-split {resplit} long paragraphs via spaCy Pass 2")
else:
    print("  ⚠  spaCy not available — Pass 2 skipped")

# ============================================================
# SAVE
# ============================================================
df       = pd.DataFrame(all_clauses)
out_path = os.path.join(SEGMENTED_DIR, "clauses_raw.csv")
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"\n✓ Saved {len(df)} clauses → {out_path}")


# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 60)
print("SEGMENTATION REPORT")
print("=" * 60)

for domain, count in domain_stats.items():
    bar = "█" * (count // 50)
    print(f"  {domain:<20} {count:>5} clauses  {bar}")

print(f"\n  Total clauses    : {len(df)}")
print(f"  Avg words/clause : {df['word_count'].mean():.1f}")
print(f"  Min / Max words  : {df['word_count'].min()} / {df['word_count'].max()}")

print(f"\n  Pattern breakdown:")
for label, cnt in df['pattern_label'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"    {label:<28} {cnt:>5}  ({pct:.1f}%)")

print("\n── Sanity Checks ──────────────────────────────────────")
checks = [
    (len(df) < 2000,               "⚠  Low total — segmenter merging too much. Lower MIN_WORDS."),
    (len(df) > 10000,              "⚠  High total — over-splitting. Raise MIN_WORDS."),
    (df['word_count'].mean() < 10, "⚠  Avg too low — many fragments. Raise MIN_WORDS to 10."),
    (df['word_count'].mean() > 50, "⚠  Avg too high — clauses under-split."),
]
any_warn = False
for condition, msg in checks:
    if condition:
        print(f"  {msg}")
        any_warn = True
if not any_warn:
    print("  ✓ All checks passed")

