import os
import re
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "segmented")
INPUT_PATH    = os.path.join(SEGMENTED_DIR, "clauses_raw.csv")
OUTPUT_PATH   = os.path.join(SEGMENTED_DIR, "clauses_clean.csv")

JACCARD_THRESHOLD = 0.85  # token overlap threshold for near-duplicates
MIN_ALPHA_RATIO   = 0.55  # min alphabetic character ratio
MAX_NUMBER_RATIO  = 0.40  # max numeric token ratio

print("=" * 60)
print("CLEAN + DEDUPLICATE")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"\nLoaded {len(df)} clauses")
print("\nBefore cleaning:")
for domain, cnt in df['domain'].value_counts().items():
    print(f"  {domain:<15} {cnt:>5}")


# ============================================================
# EXACT DEDUPLICATION
# ============================================================
before    = len(df)
df['_norm'] = df['raw_text'].str.lower().str.strip()
df          = df.drop_duplicates(subset='_norm').drop(columns=['_norm'])
print(f"\n[4A] Exact dedup   : -{before - len(df):>4} removed  →  {len(df)} remain")


# ============================================================
# NOISE REMOVAL
# ============================================================
def is_clean(text):
    tokens = text.split()
    if not tokens:
        return False

    # Must be mostly alphabetic
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False

    # Must not be number-heavy
    num_tokens = sum(1 for t in tokens if re.fullmatch(r'[\d\.\,\%\$\-\/\(\)]+', t))
    if num_tokens / len(tokens) > MAX_NUMBER_RATIO:
        return False

    # No repeated character garbage (e.g. "_ _ _ _" or "-------")
    if re.search(r'(.)\1{6,}', text):
        return False

    # Must contain at least one meaningful verb
    if not re.search(
        r'\b(is|are|was|were|will|shall|must|may|should|have|has|had|can|could'
        r'|would|do|does|did|be|been|require|provide|ensure|comply|result'
        r'|constitute|mean|include|apply|define|allow|prohibit|subject|contain'
        r'|establish|determine|render|cause|effect|make|give|take|use|set)\w*\b',
        text, re.IGNORECASE
    ):
        return False

    return True

before = len(df)
df     = df[df['raw_text'].apply(is_clean)].reset_index(drop=True)
print(f"[4B] Noise removal : -{before - len(df):>4} removed  →  {len(df)} remain")


# ============================================================
# NEAR-DUPLICATE REMOVAL (Jaccard on token sets)
# Runs per domain to keep it fast — O(n²) within domain only
# ============================================================
def tokenset(text):
    return set(re.findall(r'\b\w+\b', text.lower()))

before     = len(df)
keep_flags = [True] * len(df)
tokensets  = [tokenset(t) for t in df['raw_text']]

for domain in df['domain'].unique():
    idx = df.index[df['domain'] == domain].tolist()
    for i in tqdm(range(len(idx)), desc=f"  Near-dedup [{domain:<10}]", leave=False):
        if not keep_flags[idx[i]]:
            continue
        for j in range(i + 1, len(idx)):
            if not keep_flags[idx[j]]:
                continue
            a, b = tokensets[idx[i]], tokensets[idx[j]]
            if len(a & b) / max(len(a | b), 1) >= JACCARD_THRESHOLD:
                keep_flags[idx[j]] = False

df = df[keep_flags].reset_index(drop=True)
print(f"[4C] Near-dedup    : -{before - len(df):>4} removed  →  {len(df)} remain")


# ============================================================
# RE-INDEX AND SAVE
# ============================================================
df = df.sort_values(['domain', 'source_doc']).reset_index(drop=True)
df['clause_id'] = [f"C{i+1:05d}" for i in range(len(df))]

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\n✓ Saved → {OUTPUT_PATH}")


# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 FINAL REPORT")
print("=" * 60)
print(f"\n  Total clean clauses : {len(df)}")
print(f"  Avg words/clause    : {df['word_count'].mean():.1f}")

print(f"\n  Domain breakdown:")
for domain, cnt in df['domain'].value_counts().items():
    pct = cnt / len(df) * 100
    bar = "█" * (cnt // 30)
    print(f"    {domain:<15} {cnt:>5}  ({pct:.1f}%)  {bar}")

print(f"\n  Pattern breakdown:")
for label, cnt in df['pattern_label'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"    {label:<28} {cnt:>4}  ({pct:.1f}%)")

print(f"\n  Unique source docs  : {df['source_doc'].nunique()}")

