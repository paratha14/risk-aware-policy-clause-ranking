import os
import re
import pandas as pd
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm")

# Keep domain-relevant stopwords that carry risk meaning
# e.g. "not", "must", "shall" — these matter for risk detection
KEEP_WORDS = {
    "must", "shall", "may", "should", "will", "not", "no",
    "without", "unless", "except", "never", "only", "immediately",
    "prohibited", "required", "mandatory", "forbidden"
}

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "segmented")
INPUT_PATH    = os.path.join(SEGMENTED_DIR, "clauses_clean.csv")
OUTPUT_PATH   = os.path.join(SEGMENTED_DIR, "clauses_preprocessed.csv")

print("=" * 60)
print("TEXT PREPROCESSING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"\nLoaded {len(df)} clauses")


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================
def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline for a single clause.
    Returns lemmatized, stopword-filtered clean string.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 3. Remove special characters — keep letters, digits, spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. spaCy: tokenize, remove stopwords, lemmatize
    doc    = nlp(text)
    tokens = []
    for token in doc:
        # Keep if: in our KEEP_WORDS, or not a stopword and not punctuation
        if token.text in KEEP_WORDS:
            tokens.append(token.text)          # keep original form
        elif (not token.is_stop
              and not token.is_punct
              and not token.is_space
              and len(token.lemma_) > 1):
            tokens.append(token.lemma_)        # lemmatized form

    return " ".join(tokens)


# ============================================================
# APPLY TO ALL CLAUSES
# ============================================================
print("\nPreprocessing clauses...")
tqdm.pandas(desc="  Processing")
# Filter code snippets before preprocessing
CODE_SIGNALS = ['function(', 'function (', 'getElementById',
                'querySelector', 'console.', '.push(', '.map(',
                '// ', '<?php', 'var ', 'const ', 'let ']

def has_code(text):
    tl = text.lower()
    return any(sig.lower() in tl for sig in CODE_SIGNALS)

before_code = len(df)
df = df[~df['raw_text'].apply(has_code)].reset_index(drop=True)
print(f"  Removed {before_code - len(df)} code snippet clauses")

df['clean_text'] = df['raw_text'].progress_apply(preprocess)

# Flag empty results (shouldn't happen but good to check)
empty = df['clean_text'].str.strip().eq('').sum()
if empty > 0:
    print(f"  ⚠  {empty} clauses produced empty clean_text — removing")
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

# Add token count column (useful for feature engineering in Step 7)
df['token_count'] = df['clean_text'].apply(lambda x: len(x.split()))


# ============================================================
# SAVE
# ============================================================
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\n✓ Saved → {OUTPUT_PATH}")


# ============================================================
# REPORT
# ============================================================
print(f"\n  Total clauses       : {len(df)}")
print(f"  Avg raw words       : {df['word_count'].mean():.1f}")
print(f"  Avg clean tokens    : {df['token_count'].mean():.1f}")
print(f"  Compression ratio   : {df['token_count'].mean()/df['word_count'].mean():.2f}  (clean/raw)")

print(f"\n  Domain breakdown:")
for domain, cnt in df['domain'].value_counts().items():
    print(f"    {domain:<15} {cnt:>5}")

print(f"\n  Sample output:")
for _, row in df.sample(3, random_state=42).iterrows():
    print(f"\n  [{row['domain']}]")
    print(f"  RAW   : {row['raw_text'][:100]}")
    print(f"  CLEAN : {row['clean_text'][:100]}")

print(f"\n  Columns in output CSV:")
for col in df.columns:
    print(f"    {col}")
