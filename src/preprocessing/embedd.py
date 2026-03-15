import os, pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SEG          = os.path.join(PROJECT_ROOT, "data", "segmented")
EMB          = os.path.join(PROJECT_ROOT, "data", "embeddings")
os.makedirs(EMB, exist_ok=True)

df = pd.read_csv(os.path.join(SEG, "clauses_features.csv"))
print(f"Loaded {len(df)} clauses")

# ── TF-IDF on clean_text ─────────────────────────────────────
print("\n[1/3] TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                        sublinear_tf=True, min_df=2)
X_tfidf = tfidf.fit_transform(df["clean_text"].fillna(""))

save_npz(os.path.join(EMB, "tfidf_matrix.npz"), X_tfidf)
with open(os.path.join(EMB, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
print(f"  ✓ TF-IDF shape : {X_tfidf.shape}")

# ── SBERT on raw_text ────────────────────────────────────────
print("\n[2/3] SBERT embeddings (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    df["raw_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
np.save(os.path.join(EMB, "sbert_embeddings.npy"), embeddings)
print(f"  ✓ SBERT shape  : {embeddings.shape}")

# ── Label encoding ───────────────────────────────────────────
print("\n[3/3] Encoding labels...")
le = LabelEncoder()
y  = le.fit_transform(df["risk_label"])
np.save(os.path.join(EMB, "labels_encoded.npy"), y)
with open(os.path.join(EMB, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
print(f"  ✓ Classes : {list(le.classes_)}")
print(f"  ✓ Labels  : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Save clause IDs for alignment ────────────────────────────
df[["clause_id","risk_label","domain"]].to_csv(
    os.path.join(EMB, "embedding_index.csv"), index=False)

print(f"\n✓ All saved → {EMB}/")
print("  tfidf_matrix.npz      — sparse TF-IDF (3000 features)")
print("  tfidf_vectorizer.pkl  — fitted vectorizer for inference")
print("  sbert_embeddings.npy  — dense SBERT (384 dims)")
print("  labels_encoded.npy    — integer labels")
print("  label_encoder.pkl     — LabelEncoder for inverse transform")
print("  embedding_index.csv   — clause_id alignment file")