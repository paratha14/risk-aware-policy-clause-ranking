import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "segmented")
INPUT_PATH    = os.path.join(SEGMENTED_DIR, "clauses_preprocessed.csv")
OUTPUT_PATH   = os.path.join(SEGMENTED_DIR, "clauses_labeled.csv")

print("=" * 60)
print("RISK LABELING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} clauses")


# ============================================================
# KEYWORD LISTS
# ============================================================

MODAL_VERBS = {
    "must": 0.9, "shall": 0.9, "obliged": 0.9, "obligated": 0.9,
    "mandated": 0.9, "will": 0.8, "required": 0.8,
    "should": 0.6, "ought": 0.6, "expected": 0.6, "necessary": 0.6,
    "may": 0.3, "can": 0.3, "could": 0.3, "might": 0.2,
    "allowed": 0.3, "permitted": 0.3, "encouraged": 0.1,
    "recommended": 0.1, "suggested": 0.1, "optional": 0.1,
}

CONSEQUENCE_KEYWORDS = {
    # Critical (0.85 - 1.0)
    "termination": 1.0, "terminated": 1.0, "terminate": 1.0,
    "dismissal": 1.0, "dismissed": 1.0, "dismiss": 1.0,
    "expulsion": 1.0, "expelled": 1.0, "expel": 1.0,
    "disqualification": 1.0, "disqualified": 1.0,
    "imprisonment": 1.0, "prosecution": 1.0, "prosecuted": 1.0,
    "criminal": 0.95, "conviction": 1.0, "convicted": 1.0,
    "revocation": 0.95, "revoked": 0.95,
    "injunction": 0.95, "litigation": 0.95, "lawsuit": 0.95,
    "legal action": 1.0, "law enforcement": 1.0,
    "indemnify": 0.9, "indemnification": 0.9, "indemnif": 0.9,
    "damages": 0.85, "penalty": 0.85, "penalties": 0.85,
    "fine": 0.85, "fined": 0.85,
    "sanction": 0.85, "sanctions": 0.85,
    "banned": 0.9, "ban": 0.9, "barred": 0.9,
    "debarred": 0.9, "blacklisted": 0.9,
    "void": 0.9, "liable": 0.85, "liability": 0.85,
    "breach": 0.8, "default": 0.8,
    "infringement": 0.85, "infringe": 0.85,
    "misappropriation": 0.9,
    "arbitration": 0.85, "arbitrate": 0.85,
    "injunctive": 0.9, "equitable relief": 0.9,
    "liquidated damages": 0.95, "consequential damages": 0.9,
    "specific performance": 0.85,
    "material breach": 0.8, "material default": 0.8,
    # High (0.55 - 0.84)
    "suspension": 0.75, "suspended": 0.75,
    "probation": 0.7, "disciplinary": 0.7,
    "investigation": 0.65, "investigated": 0.65,
    "formal warning": 0.7, "written warning": 0.7,
    "prohibited": 0.7, "prohibit": 0.7,
    "restriction": 0.65, "restricted": 0.65,
    "withhold": 0.65, "withheld": 0.65,
    "loss of": 0.7, "forfeiture": 0.75, "forfeit": 0.7,
    "corrective action": 0.65, "performance plan": 0.6,
    "remedies": 0.65, "remedy": 0.6,
    "notice of breach": 0.75,
    "not permitted": 0.55, "not allowed": 0.55,
    # Medium (0.25 - 0.54)
    "warning": 0.5, "warned": 0.5,
    "deduction": 0.45, "deducted": 0.45,
    "monitor": 0.4, "monitored": 0.4,
    "hold": 0.4, "freeze": 0.4,
    "reprimand": 0.5, "caution": 0.45,
    "obligation": 0.45, "obligations": 0.45,
    "covenant": 0.45, "covenants": 0.45,
    "warranty": 0.45, "warranties": 0.45,
    "represent and warrant": 0.5,
    "comply": 0.4, "compliance": 0.4,
    # Low (0.05 - 0.24)
    "inform": 0.2, "remind": 0.15, "advise": 0.15,
    "note": 0.1, "consider": 0.1, "encourage": 0.1,
    "recommend": 0.1, "suggest": 0.1, "guidance": 0.1,
    "discretion": 0.2, "acknowledge": 0.15,
    "agrees": 0.2, "agree": 0.2,
}

CONDITIONAL_KEYWORDS = {
    # Strong (0.7 - 1.0) — directly introduce consequences
    "failure to": 1.0, "failure of": 1.0,
    "non-compliance": 1.0, "noncompliance": 1.0,
    "in violation": 1.0, "violation of": 1.0,
    "breach of": 1.0, "in breach": 0.95,
    "liable for": 0.95, "liable to": 0.9,
    "result in": 0.85, "results in": 0.85,
    "constitute": 0.8, "constitutes": 0.8,
    "give rise to": 0.85, "in the event of": 0.85,
    "without authorization": 0.85, "without consent": 0.85,
    "without notice": 0.85, "strictly prohibited": 0.95,
    "is prohibited": 0.9, "must not": 0.85, "shall not": 0.85,
    "zero tolerance": 1.0,
    "upon breach": 0.8, "upon default": 0.75,
    "without prior written": 0.8,
    "if terminated": 0.75, "if breached": 0.75,
    "upon termination": 0.75,
    # Medium (0.4 - 0.69)
    "unless": 0.5, "provided that": 0.6,
    "may result": 0.6, "could result": 0.6,
    "in the event": 0.6, "notwithstanding": 0.55,
    "in case of": 0.55, "subject to approval": 0.5,
    "unless otherwise": 0.55,
    "except as": 0.45, "without written": 0.65,
    # Low (0.1 - 0.39) — intentionally low to not inflate scores
    "where possible": 0.15, "at discretion": 0.2,
    "where applicable": 0.15, "as appropriate": 0.15,
    "may be waived": 0.1, "where relevant": 0.15,
    # REMOVED "subject to" — fires on "subject to the terms" which is NOT a risk signal
    # REMOVED "immediately" — too broad, appears in Low clauses
}

AMPLIFIERS_UP = {
    "immediate": 1.3, "immediately": 1.3,
    "permanent": 1.3, "permanently": 1.3,
    "gross": 1.3, "gross misconduct": 1.35,
    "serious": 1.25, "severe": 1.25, "significant": 1.2,
    "strict": 1.2, "strictly": 1.2,
    "automatic": 1.25, "automatically": 1.25,
    "mandatory": 1.2, "without exception": 1.3,
    "regardless": 1.15, "unconditional": 1.2,
    "per violation": 1.25, "each violation": 1.25,
    "willful": 1.25, "wilful": 1.25,
    "deliberate": 1.2, "repeated": 1.2,
}

AMPLIFIERS_DOWN = {
    "minor": 0.7, "minimal": 0.75, "incidental": 0.75,
    "reasonable": 0.8, "where practicable": 0.7,
    "at discretion": 0.75, "may be waived": 0.65,
    "generally": 0.85, "typically": 0.85,
    "where possible": 0.75, "informal": 0.7,
    "verbal": 0.75, "first offence": 0.8,
    "first offense": 0.8, "advisory": 0.65,
    "informational": 0.6, "optional": 0.6,
    "encouraged": 0.65, "recommended": 0.7,
}

W_MODAL, W_CONSEQUENCE, W_CONDITIONAL = 0.25, 0.50, 0.25


# ============================================================
# PASS 1 — TOKEN SCORING
# ============================================================
def pass1_score(text: str) -> float:
    text_lower = text.lower()
    tokens     = [t.text.lower() for t in nlp(text_lower)]

    modal_hits  = [MODAL_VERBS[t] for t in tokens if t in MODAL_VERBS]
    modal_score = max(modal_hits) if modal_hits else 0.0

    cons_score = 0.0
    for phrase, weight in sorted(CONSEQUENCE_KEYWORDS.items(),
                                  key=lambda x: len(x[0]), reverse=True):
        if phrase in text_lower:
            cons_score = max(cons_score, weight)

    cond_score = 0.0
    for phrase, weight in sorted(CONDITIONAL_KEYWORDS.items(),
                                  key=lambda x: len(x[0]), reverse=True):
        if phrase in text_lower:
            cond_score = max(cond_score, weight)

    # Modal alone = informational → reduce heavily so it scores Low
    if cons_score == 0.0 and cond_score == 0.0:
        modal_score = modal_score * 0.15

    score = (modal_score * W_MODAL +
             cons_score  * W_CONSEQUENCE +
             cond_score  * W_CONDITIONAL)

    # Amplifiers
    amp = 1.0
    for phrase, factor in AMPLIFIERS_UP.items():
        if phrase in text_lower:
            amp = max(amp, factor)
    for phrase, factor in AMPLIFIERS_DOWN.items():
        if phrase in text_lower:
            amp = min(amp, factor)

    # Consequence override for truly critical keywords
    if cons_score >= 0.95:
        score = max(score, 0.75)

    return min(score * amp, 1.0)


# ============================================================
# PASS 2 — N-GRAM VERIFICATION
# ============================================================
HIGH_RISK_SEEDS = [
    "termination of employment", "immediate dismissal", "criminal prosecution",
    "legal action", "subject to termination", "gross misconduct",
    "result in dismissal", "result in expulsion", "disqualified from",
    "referred to law enforcement", "without prior notice",
    "shall be terminated", "may be prosecuted", "liable for damages",
    "regulatory fine", "enforcement action", "permanent exclusion",
    "academic misconduct", "failure to comply", "breach of contract",
    "in violation of", "subject to disciplinary", "loss of privileges",
    "suspension without pay", "written warning issued",
    "zero tolerance policy", "data breach notification",
    "strictly prohibited", "must not disclose", "shall not be",
    "non-compliance may result", "violation of this policy",
]

print("\nBuilding n-gram vocabulary...")
vectorizer = CountVectorizer(
    ngram_range=(1, 3), min_df=2,
    max_features=8000, token_pattern=r'\b[a-z][a-z]+\b'
)
vectorizer.fit(df['raw_text'].str.lower())
vocab = set(vectorizer.vocabulary_.keys())

high_risk_ngrams = set()
for seed in HIGH_RISK_SEEDS:
    seed_lower = seed.lower()
    for ngram in vocab:
        if ngram in seed_lower or seed_lower in ngram:
            high_risk_ngrams.add(ngram)

for phrase in CONSEQUENCE_KEYWORDS:
    if phrase in vocab and CONSEQUENCE_KEYWORDS[phrase] >= 0.7:
        high_risk_ngrams.add(phrase)
for phrase in CONDITIONAL_KEYWORDS:
    if phrase in vocab and CONDITIONAL_KEYWORDS[phrase] >= 0.7:
        high_risk_ngrams.add(phrase)

print(f"  High-risk n-grams identified: {len(high_risk_ngrams)}")


def pass2_check(text: str, p1: float) -> float:
    # FIX: if pass1 scored 0.0 (truly informational), don't escalate
    # Only escalate clauses that pass1 already detected SOME risk signal
    if p1 == 0.0:
        return 0.0

    text_lower = text.lower()
    found = [ng for ng in high_risk_ngrams if ng in text_lower]
    if not found:
        return p1

    ng_scores = []
    for ng in found:
        if ng in CONSEQUENCE_KEYWORDS:
            ng_scores.append(CONSEQUENCE_KEYWORDS[ng])
        elif ng in CONDITIONAL_KEYWORDS:
            ng_scores.append(CONDITIONAL_KEYWORDS[ng] * 0.8)
        # FIX: removed default 0.6 — only score if in known dicts

    if not ng_scores:
        return p1

    ng_max = max(ng_scores)
    if ng_max > p1 + 0.15:
        return (p1 + ng_max) / 2
    return p1


# ============================================================
# LABEL MAPPING
# ============================================================
def score_to_label(score: float) -> str:
    if score >= 0.75:   return "Critical"
    elif score >= 0.50: return "High"
    elif score >= 0.10: return "Medium"
    else:               return "Low"


# ============================================================
# APPLY BOTH PASSES
# ============================================================
print("\nRunning Pass 1 (token scoring)...")
p1_scores = [pass1_score(str(t)) for t in tqdm(df['raw_text'], desc="  Pass 1")]

print("Running Pass 2 (n-gram verification)...")
final_scores = [pass2_check(str(t), p1)
                for t, p1 in tqdm(zip(df['raw_text'], p1_scores),
                                  total=len(df), desc="  Pass 2")]

df['risk_score'] = [round(s, 4) for s in final_scores]
df['risk_label'] = df['risk_score'].apply(score_to_label)

# Quick distribution check before balancing
print("\nPre-balance distribution:")
for label in ["Critical", "High", "Medium", "Low"]:
    cnt = (df['risk_label'] == label).sum()
    print(f"  {label:<12} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")


# ============================================================
# BALANCING CLASSES
# ============================================================
print("\nBalancing risk classes...")

TARGET   = 1000
balanced = []

for label in ["Critical", "High", "Medium", "Low"]:
    subset = df[df['risk_label'] == label]
    count  = len(subset)

    if count == 0:
        print(f"  ⚠  {label}: 0 clauses — check scoring thresholds")
        continue
    elif count >= TARGET:
        sampled = subset.nlargest(TARGET, 'risk_score')
        print(f"  {label:<12} {count:>5} → {len(sampled)} (undersampled)")
    else:
        repeats = (TARGET // count) + 1
        sampled = pd.concat([subset] * repeats, ignore_index=True)
        sampled = sampled.sample(TARGET, random_state=42)
        print(f"  {label:<12} {count:>5} → {len(sampled)} (oversampled)")

    balanced.append(sampled)

df_balanced = pd.concat(balanced, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced['clause_id'] = [f"C{i+1:05d}" for i in range(len(df_balanced))]

BALANCED_PATH = OUTPUT_PATH.replace("clauses_labeled", "clauses_balanced")
df_balanced.to_csv(BALANCED_PATH, index=False, encoding="utf-8")
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n✓ Saved labeled   → {OUTPUT_PATH}")
print(f"✓ Saved balanced  → {BALANCED_PATH}")


# ============================================================
# REPORT
# ============================================================

print(f"\n  Total labeled clauses : {len(df)}")
print(f"\n  Risk score stats:")
print(f"    Min / Max : {df['risk_score'].min():.4f} / {df['risk_score'].max():.4f}")
print(f"    Mean / Std: {df['risk_score'].mean():.3f} / {df['risk_score'].std():.3f}")

print(f"\n  Risk label distribution:")
for label in ["Critical", "High", "Medium", "Low"]:
    cnt = (df['risk_label'] == label).sum()
    bar = "█" * (cnt // 30)
    print(f"    {label:<12} {cnt:>5}  ({cnt/len(df)*100:.1f}%)  {bar}")

print(f"\n  Risk distribution by domain:")
pivot = df.groupby(['domain','risk_label']).size().unstack(fill_value=0)
for col in ["Critical","High","Medium","Low"]:
    if col not in pivot.columns:
        pivot[col] = 0
print(pivot[["Critical","High","Medium","Low"]].to_string())

print(f"\n  Sample labeled clauses:")
for label in ["Critical", "High", "Medium", "Low"]:
    subset = df[df['risk_label'] == label]
    if len(subset) == 0:
        print(f"\n  [{label}] — no clauses")
        continue
    row = subset.sample(1, random_state=42).iloc[0]
    print(f"\n  [{label}] score={row['risk_score']:.4f}")
    print(f"  {row['raw_text'][:120]}")

print(f"\n  Balanced distribution:")
for label in ["Critical", "High", "Medium", "Low"]:
    cnt = (df_balanced['risk_label'] == label).sum()
    bar = "█" * (cnt // 40)
    print(f"    {label:<12} {cnt:>5}  ({cnt/len(df_balanced)*100:.1f}%)  {bar}")

print(f"\n  Total balanced : {len(df_balanced)}")
