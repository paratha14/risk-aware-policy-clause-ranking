import os
import re
import json
import pdfplumber
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../src/
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)                 
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw_docs")
EXTRACTED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(EXTRACTED_DIR, exist_ok=True)

print("=" * 60)
print("STEP 2 — PDF → TEXT EXTRACTION")
print("=" * 60)


# ============================================================
# HELPER: Extract text from a single PDF
# ============================================================
def extract_pdf(filepath: str) -> str:
    """
    Extract all text from a PDF using pdfplumber.
    Falls back to empty string on failure.
    Cleans up common PDF artifacts automatically.
    """
    try:
        full_text = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)

        raw = "\n".join(full_text)
        return clean_extracted_text(raw)

    except Exception as e:
        print(f"      PDF extraction failed for {os.path.basename(filepath)}: {e}")
        return ""


# ============================================================
# HELPER: Extract text from a plain .txt file
# ============================================================
def extract_txt(filepath: str) -> str:
    """Read plain text file with encoding fallback."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return clean_extracted_text(f.read())
        except UnicodeDecodeError:
            continue
    return ""


# ============================================================
# HELPER: Clean raw extracted text
# ============================================================
def clean_extracted_text(text: str) -> str:
    """
    Remove common PDF noise while preserving structure.
    """
    # Remove page numbers (standalone digits on a line)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Remove headers/footers (repeated short lines)
    lines = text.split('\n')
    line_freq = {}
    for line in lines:
        stripped = line.strip()
        if 5 < len(stripped) < 80:
            line_freq[stripped] = line_freq.get(stripped, 0) + 1

    # Drop lines that repeat 3+ times (typical header/footer)
    repeated = {l for l, c in line_freq.items() if c >= 3}
    lines = [l for l in lines if l.strip() not in repeated]
    text = '\n'.join(lines)

    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


# ============================================================
# HELPER: Detect file type and route to correct extractor
# ============================================================
def extract_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in [".txt", ".text"]:
        return extract_txt(filepath)
    else:
        return ""   # skip unsupported types


# ============================================================
# MAIN: Walk all domain folders and extract
# ============================================================
manifest = []   # track every file processed
stats    = {}  # per-domain counts

domains = [d for d in os.listdir(RAW_DIR)
           if os.path.isdir(os.path.join(RAW_DIR, d))]

for domain in sorted(domains):
    domain_in  = os.path.join(RAW_DIR, domain)
    domain_out = os.path.join(EXTRACTED_DIR, domain)
    os.makedirs(domain_out, exist_ok=True)

    files = [f for f in os.listdir(domain_in)
             if os.path.splitext(f)[1].lower() in [".pdf", ".txt", ".text"]]

    print(f"\n[{domain.upper()}] — {len(files)} files")

    success, skipped = 0, 0

    for fname in tqdm(files, desc=f"  Extracting {domain}"):
        src_path  = os.path.join(domain_in, fname)
        base_name = os.path.splitext(fname)[0]
        out_path  = os.path.join(domain_out, base_name + ".txt")

        text = extract_file(src_path)

        if len(text.split()) < 50:
            # Too short — likely a failed extract or empty doc
            skipped += 1
            status = "skipped_too_short"
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            success += 1
            status = "ok"

        manifest.append({
            "domain"    : domain,
            "filename"  : fname,
            "source_path" : src_path,
            "output_path" : out_path if status == "ok" else "",
            "word_count": len(text.split()),
            "char_count": len(text),
            "status"    : status,
        })

    stats[domain] = {"success": success, "skipped": skipped}
    print(f"    ✓ {success} extracted  |  {skipped} skipped")


manifest_path = os.path.join(PROJECT_ROOT, "data", "extraction_manifest.csv")
df_manifest = pd.DataFrame(manifest)
df_manifest.to_csv(manifest_path, index=False)
print(f"\n  ✓ Manifest saved → {manifest_path}")


# ============================================================
# QUALITY REPORT
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

total_ok   = sum(v["success"] for v in stats.values())
total_skip = sum(v["skipped"] for v in stats.values())
total_words= df_manifest[df_manifest["status"]=="ok"]["word_count"].sum()

for domain, s in stats.items():
    print(f"  {domain:<20}  extracted: {s['success']:>3}  |  skipped: {s['skipped']:>2}")

print(f"\n  Total documents extracted : {total_ok}")
print(f"  Total documents skipped   : {total_skip}")
print(f"  Total words across corpus : {total_words:,}")
print(f"\n  Extracted text saved to   : {EXTRACTED_DIR}/")

# Quick sanity checks
print("\n── Sanity Checks ──────────────────────────────────────")
df_ok = df_manifest[df_manifest["status"] == "ok"]

if len(df_ok) == 0:
    print("  ✗ No files extracted — check your raw_docs folder")
else:
    avg_words = df_ok["word_count"].mean()
    min_words = df_ok["word_count"].min()
    max_words = df_ok["word_count"].max()
    print(f"  Avg words per doc : {avg_words:,.0f}")
    print(f"  Min words per doc : {min_words:,}")
    print(f"  Max words per doc : {max_words:,}")

    if avg_words < 200:
        print("  ⚠  Average word count low — check PDF extraction quality")
    else:
        print("  ✓  Word counts look healthy")

    domain_counts = df_ok["domain"].value_counts()
    print("\n  Docs per domain:")
    for domain, count in domain_counts.items():
        print(f"    {domain:<20} {count}")
