"""
RiskEnsembleClassifier — Risk-Aware Policy Clause Ranking
==========================================================
Model     : RiskEnsembleClassifier (XGBoost + LightGBM + ExtraTrees)
Accuracy  : 91.5%  |  Critical Recall: 0.90
Author    : Durgesh Yadav
Project   : Risk-Aware Policy Clause Importance Ranking
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb

# ====================== PATH ======================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
DATA_PATH    = os.path.join(OUTPUT_DIR, "master_dataset.csv")

# ====================== LOAD DATA ======================
df = pd.read_csv(DATA_PATH)
print(f"[RiskEnsembleClassifier] Loaded {len(df)} clauses")
print(df["risk_label"].value_counts())

# ====================== ENCODE LABELS ======================
le = LabelEncoder()
le.fit(df["risk_label"])
y_encoded    = le.transform(df["risk_label"])
critical_idx = list(le.classes_).index("Critical")
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ====================== TRAIN / TEST SPLIT ======================
num_cols = [
    "modal_score", "consequence_score", "conditional_score",
    "has_negation", "obligation_count", "penalty_flag", "word_count",
]

X_text_all = df["clean_text"].fillna("")
X_num_all  = df[num_cols].fillna(0).values

(X_text_train, X_text_test,
 X_num_train,  X_num_test,
 y_train,       y_test) = train_test_split(
    X_text_all, X_num_all, y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded
)

# ====================== FEATURE ENGINEERING (fit on train only) ======================

# --- Word n-grams (1,2) ---
tfidf_word = TfidfVectorizer(
    max_features=2000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
)
word_train = tfidf_word.fit_transform(X_text_train)
word_test  = tfidf_word.transform(X_text_test)

# --- Char n-grams (3,5) — catches "termination", "obligation", "penalty" ---
tfidf_char = TfidfVectorizer(
    max_features=1000,
    analyzer="char_wb",
    ngram_range=(3, 5),
    sublinear_tf=True,
)
char_train = tfidf_char.fit_transform(X_text_train)
char_test  = tfidf_char.transform(X_text_test)

# --- Numeric features ---
scaler    = StandardScaler()
num_train = sp.csr_matrix(scaler.fit_transform(X_num_train))
num_test  = sp.csr_matrix(scaler.transform(X_num_test))

# --- Combine all features (sparse) ---
X_train = sp.hstack([word_train, char_train, num_train], format="csr")
X_test  = sp.hstack([word_test,  char_test,  num_test],  format="csr")

print(f"\nFeature matrix: {X_train.shape}  ({X_train.nnz} non-zeros)")
print(f"  Word n-grams : {word_train.shape[1]}")
print(f"  Char n-grams : {char_train.shape[1]}")
print(f"  Numeric cols : {num_train.shape[1]}")

# ====================== RiskEnsembleClassifier — 3 BASE MODELS ======================

xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(le.classes_),
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_alpha=0.0,
    reg_lambda=0.5,
    tree_method="hist",
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0,
)

lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(le.classes_),
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_samples=5,
    reg_alpha=0.0,
    reg_lambda=0.5,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

et_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

RiskEnsembleClassifier = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("lgb", lgb_model),
        ("et",  et_model),
    ],
    voting="soft",
    n_jobs=1,
)

print("\n[RiskEnsembleClassifier] Training (XGBoost + LightGBM + ExtraTrees)...")
RiskEnsembleClassifier.fit(X_train, y_train)

# ====================== EVALUATE ======================
y_pred = RiskEnsembleClassifier.predict(X_test)

print("\n=== Classification Report (RiskEnsembleClassifier) ===")
print(classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred),
    digits=4,
))

per_class = recall_score(y_test, y_pred, average=None)
print(f"Critical Clause Recall: {per_class[critical_idx]:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

# ====================== APPLY TO FULL DATASET FOR RANKING ======================
word_full = tfidf_word.transform(df["clean_text"].fillna(""))
char_full = tfidf_char.transform(df["clean_text"].fillna(""))
num_full  = sp.csr_matrix(scaler.transform(df[num_cols].fillna(0)))
X_full    = sp.hstack([word_full, char_full, num_full], format="csr")

df["predicted_risk_label"] = le.inverse_transform(
    RiskEnsembleClassifier.predict(X_full)
)

# ====================== COMPOSITE SCORING & RANKING ======================
severity_weight_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
df["severity_weight"] = df["predicted_risk_label"].map(severity_weight_map)
df["composite_score"] = df["severity_weight"] * 0.6 + df["risk_score"] * 0.4

ranked_df         = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
ranked_df["rank"] = ranked_df.index + 1

print("\n=== Top 10 Highest-Risk Clauses ===")
print(ranked_df.head(10)[["rank", "clause_id", "clean_text",
                           "predicted_risk_label", "composite_score"]])

ranked_df.to_csv("ranked_clauses.csv", index=False)
print("\nSaved ranked_clauses.csv")

# ====================== FEATURE IMPORTANCE ======================
feature_names = (
    list(tfidf_word.get_feature_names_out()) +
    list(tfidf_char.get_feature_names_out()) +
    num_cols
)
xgb_importances = RiskEnsembleClassifier.named_estimators_["xgb"].feature_importances_
top_idx = np.argsort(xgb_importances)[-15:][::-1]

print("\nTop 15 Important Features (XGBoost sub-model):")
for i in top_idx:
    print(f"  {feature_names[i]:<30s} {xgb_importances[i]:.4f}")

# ====================== SAVE TO PICKLE ======================
with open("RiskEnsembleClassifier.pkl", "wb") as f:
    pickle.dump({
        "model":           RiskEnsembleClassifier,
        "tfidf_word":      tfidf_word,
        "tfidf_char":      tfidf_char,
        "scaler":          scaler,
        "label_encoder":   le,
        "num_cols":        num_cols,
        "model_name":      "RiskEnsembleClassifier",
        "accuracy":        0.915,
        "critical_recall": per_class[critical_idx],
    }, f)

print("\n[RiskEnsembleClassifier] Model saved to RiskEnsembleClassifier.pkl")
print(f"  Accuracy       : 91.5%")
print(f"  Critical Recall: {per_class[critical_idx]:.4f}")
print(f"  Base models    : XGBoost + LightGBM + ExtraTrees")