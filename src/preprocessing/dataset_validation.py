import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SEG          = os.path.join(PROJECT_ROOT, "data", "segmented")
EMB          = os.path.join(PROJECT_ROOT, "data", "embeddings")

df         = pd.read_csv(os.path.join(SEG, "clauses_features.csv"))
embeddings = np.load(os.path.join(EMB, "sbert_embeddings.npy"))
tfidf      = load_npz(os.path.join(EMB, "tfidf_matrix.npz"))
labels     = np.load(os.path.join(EMB, "labels_encoded.npy"))

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

log("=" * 60)
log("DATASET VALIDATION REPORT")
log("=" * 60)

# ── Basic stats ───────────────────────────────────────────
log("\n[1] CORPUS STATS")
log(f"  Total clauses      : {len(df)}")
log(f"  Unique source docs : {df['source_doc'].nunique()}")
log(f"  Domains            : {dict(df['domain'].value_counts())}")
log(f"  Avg word count     : {df['word_count'].mean():.1f}")
log(f"  Avg token count    : {df['token_count'].mean():.1f}")

# ── Label distribution ────────────────────────────────────
log("\n[2] LABEL DISTRIBUTION")
counts = df['risk_label'].value_counts()
for label, cnt in counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * (cnt // 40)
    log(f"  {label:<12} {cnt:>5}  ({pct:.1f}%)  {bar}")

ratio = counts.max() / counts.min()
log(f"\n  Imbalance ratio : {ratio:.2f}x  {'✓ good' if ratio < 1.5 else '⚠ check'}")

# ── Feature stats ─────────────────────────────────────────
log("\n[3] LINGUISTIC FEATURE STATS")
feat_cols = ["modal_score", "consequence_score", "conditional_score",
             "has_negation", "obligation_count", "penalty_flag"]
for col in feat_cols:
    if col in df.columns:
        log(f"  {col:<25} mean={df[col].mean():.3f}  std={df[col].std():.3f}")

# ── 4. Embedding sanity ──────────────────────────────────────
log("\n[4] EMBEDDING SANITY CHECKS")
log(f"  SBERT shape        : {embeddings.shape}")
log(f"  TF-IDF shape       : {tfidf.shape}")
log(f"  SBERT mean norm    : {np.linalg.norm(embeddings, axis=1).mean():.3f}")
log(f"  Any NaN in SBERT   : {np.isnan(embeddings).any()}")
log(f"  Labels shape       : {labels.shape}")
log(f"  Labels aligned     : {len(labels) == len(df)}")

# ── Inter-class separability ──────────────────────────────
log("\n[5] INTER-CLASS SEPARABILITY (SBERT cosine similarity)")
label_list = ["Critical", "High", "Medium", "Low"]

log("  Intra-class (higher = more cohesive):")
for label in label_list:
    idx = df[df['risk_label'] == label].index.tolist()
    if len(idx) < 2:
        continue
    sample = embeddings[idx[:100]]
    sim    = cosine_similarity(sample)
    np.fill_diagonal(sim, 0)
    log(f"    {label:<12} {sim.mean():.3f}")

log("\n  Cross-class (lower = better separation):")
for i, l1 in enumerate(label_list):
    for l2 in label_list[i+1:]:
        idx1 = df[df['risk_label'] == l1].index[:50]
        idx2 = df[df['risk_label'] == l2].index[:50]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        sim = cosine_similarity(embeddings[idx1], embeddings[idx2])
        log(f"    {l1} vs {l2:<12} {sim.mean():.3f}")

# ── Domain x label coverage ───────────────────────────────
log("\n[6] DOMAIN x LABEL COVERAGE")
pivot = df.groupby(['domain', 'risk_label']).size().unstack(fill_value=0)
for col in ["Critical", "High", "Medium", "Low"]:
    if col not in pivot.columns:
        pivot[col] = 0
log(pivot[["Critical", "High", "Medium", "Low"]].to_string())

# ── Readiness checklist ───────────────────────────────────
log("\n[7] READINESS CHECKLIST")
checks = {
    "4 risk labels present"     : df['risk_label'].nunique() == 4,
    "Min 500 per class"         : counts.min() >= 500,
    "SBERT no NaN"              : not np.isnan(embeddings).any(),
    "Embeddings aligned"        : len(labels) == len(df),
    "TF-IDF features > 1000"   : tfidf.shape[1] > 1000,
    "Linguistic features exist" : all(c in df.columns for c in feat_cols),
    "Imbalance ratio < 2x"     : ratio < 2.0,
}
all_pass = True
for check, result in checks.items():
    status = "✓" if result else "✗"
    log(f"  {status}  {check}")
    if not result:
        all_pass = False

log(f"\n  {'✓ ALL CHECKS PASSED' if all_pass else '✗ FIX THE ISSUES ABOVE'}")

# Save validation report
report_path = os.path.join(SEG, "validation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n✓ Validation report saved → {report_path}")


# ============================================================
# FINAL EXPORT
# ============================================================

master_cols = [
    "clause_id", "source_doc", "domain",
    "raw_text", "clean_text",
    "word_count", "token_count",
    "pattern_label", "risk_label", "risk_score",
    "modal_score", "consequence_score", "conditional_score",
    "has_negation", "obligation_count", "penalty_flag"
]
master_cols = [c for c in master_cols if c in df.columns]
master      = df[master_cols].copy()

master_path = os.path.join(SEG, "master_dataset.csv")
master.to_csv(master_path, index=False)
print(f"\n✓ master_dataset.csv → {master_path}")
print(f"  Shape   : {master.shape}")
print(f"  Columns : {list(master.columns)}")

print(f"\n  data/segmented/")
print(f"    master_dataset.csv       — {master.shape[0]} clauses x {master.shape[1]} columns")
print(f"    clauses_balanced.csv     — balanced training set (4000 clauses)")
print(f"    clauses_labeled.csv      — full labeled set")
print(f"    validation_report.txt    — quality report")
print(f"\n  data/embeddings/")
print(f"    sbert_embeddings.npy     — {embeddings.shape}  (dense semantic)")
print(f"    tfidf_matrix.npz         — {tfidf.shape}  (sparse lexical)")
print(f"    tfidf_vectorizer.pkl     — fitted vectorizer for inference")
print(f"    labels_encoded.npy       — integer labels {labels.shape}")
print(f"    label_encoder.pkl        — class mapping")
print(f"    embedding_index.csv      — clause_id alignment")
print(f"\n  Risk label mapping:")
print(f"    Critical → termination, criminal, damages")
print(f"    High     → suspension, breach, prohibited")
print(f"    Medium   → warning, obligation, compliance")
print(f"    Low      → definitions, advisory, informational")


