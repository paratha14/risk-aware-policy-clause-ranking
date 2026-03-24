# User Guide

## Overview

`RiskEnsembleClassifier.pkl` is a trained machine learning model that classifies policy clauses into four risk levels: **Critical**, **High**, **Medium**, and **Low**. It bundles the ensemble model along with all the fitted transformers needed to reproduce predictions.

---

## Step 1 — Generate the Pickle File

Run `RiskEnsembleClassifier.py` from inside the `Machine_Learning` folder. This trains the model on `output/master_dataset.csv` and saves the pickle file in the same directory.

> **Prerequisite:** Install dependencies first.

```bash
# From the Machine_Learning directory
pip install -r requirements.txt
python RiskEnsembleClassifier.py
```

On success you will see:
```
[RiskEnsembleClassifier] Model saved to RiskEnsembleClassifier.pkl
  Accuracy       : 91.5%
  Critical Recall: 0.90xx
  Base models    : XGBoost + LightGBM + ExtraTrees
```

---

## Step 2 — What's Inside the Pickle

The pickle file is a Python `dict` with these keys:

| Key | Description |
|---|---|
| `model` | Trained `VotingClassifier` (XGBoost + LightGBM + ExtraTrees) |
| `tfidf_word` | Fitted word n-gram `TfidfVectorizer` (max 2 000 features) |
| `tfidf_char` | Fitted char n-gram `TfidfVectorizer` (max 1 000 features) |
| `scaler` | Fitted `StandardScaler` for numeric columns |
| `label_encoder` | Fitted `LabelEncoder` → maps integers back to risk labels |
| `num_cols` | List of numeric feature column names |

---

## Step 3 — Load and Use the Model

```python
import pickle
import pandas as pd
import scipy.sparse as sp

# ── 1. Load ────────────────────────────────────────────────────────────────
with open("RiskEnsembleClassifier.pkl", "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]
tfidf_w  = bundle["tfidf_word"]
tfidf_c  = bundle["tfidf_char"]
scaler   = bundle["scaler"]
le       = bundle["label_encoder"]
num_cols = bundle["num_cols"]

# ── 2. Prepare your data ───────────────────────────────────────────────────
# Each clause must have:
#   clean_text        – pre-processed clause text (lowercase, stop-words removed)
#   modal_score       – float [0, 1]
#   consequence_score – float [0, 1]
#   conditional_score – float [0, 1]
#   has_negation      – int   0 or 1
#   obligation_count  – int   ≥ 0
#   penalty_flag      – int   0 or 1
#   word_count        – int   ≥ 1

clauses = [
    {
        "clean_text": "employee must never share trade secret proprietary data third party",
        "modal_score": 0.98, "consequence_score": 0.99, "conditional_score": 0.90,
        "has_negation": 1, "obligation_count": 4, "penalty_flag": 1, "word_count": 12
    },
    {
        "clean_text": "employee may work from home friday subject manager approval",
        "modal_score": 0.12, "consequence_score": 0.08, "conditional_score": 0.15,
        "has_negation": 0, "obligation_count": 0, "penalty_flag": 0, "word_count": 10
    },
]

df = pd.DataFrame(clauses)

# ── 3. Transform features ──────────────────────────────────────────────────
word = tfidf_w.transform(df["clean_text"])
char = tfidf_c.transform(df["clean_text"])
num  = sp.csr_matrix(scaler.transform(df[num_cols]))
X    = sp.hstack([word, char, num], format="csr")

# ── 4. Predict ─────────────────────────────────────────────────────────────
predictions = le.inverse_transform(model.predict(X))

for i, (pred, clause) in enumerate(zip(predictions, clauses)):
    print(f"Clause {i+1}: [{pred}]  →  {clause['clean_text'][:60]}...")
```

**Expected output:**
```
Clause 1: [Critical]  →  employee must never share trade secret proprietary data...
Clause 2: [Low]       →  employee may work from home friday subject manager appro...
```

---

## Risk Label Reference

| Label | Meaning |
|---|---|
| `Critical` | Severe obligation / strong penalty — requires immediate legal review |
| `High` | Significant obligation — needs close attention |
| `Medium` | Moderate obligation — routine monitoring |
| `Low` | Permissive / informational — low priority |

---

## Notes

- **sklearn version** — The pickle was generated with `scikit-learn 1.8.0`. Loading with an older version may show `InconsistentVersionWarning`; predictions still work but it is recommended to match versions.
- **clean_text format** — Feed pre-processed text (lowercased, punctuation stripped). The model was trained on such text.
- For a broader test with 20 sample clauses across all four risk levels, see `notebooks/model_testing.ipynb`.
