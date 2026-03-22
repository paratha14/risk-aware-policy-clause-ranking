"""
Clause Risk Classifier — Testing Script
========================================
Step 1: Run this ONCE to save the trained model to disk
Step 2: Use the saved model to:
  (A) Predict on a new CSV file
  (B) Type a single clause and get a prediction instantly
  (C) Error analysis — see exactly where the model is wrong
"""

import os
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# ============================================================
#  STEP 1 — SAVE YOUR TRAINED MODEL
#  Run:  python test_model.py --mode save --data "path/to/master_dataset.csv"
#  This trains the model and saves it to model_bundle.pkl
# ============================================================

def save_model(data_path):
    """Train on full data and save everything needed for prediction."""
    from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    print("Loading data...")
    df = pd.read_csv(data_path)

    num_cols = [
        "modal_score", "consequence_score", "conditional_score",
        "has_negation", "obligation_count", "penalty_flag", "word_count",
    ]

    le = LabelEncoder()
    y  = le.fit_transform(df["risk_label"])

    # Fit transformers on FULL dataset (for production use)
    tfidf_word = TfidfVectorizer(max_features=2000, stop_words="english",
                                  ngram_range=(1, 2), sublinear_tf=True)
    tfidf_char = TfidfVectorizer(max_features=1000, analyzer="char_wb",
                                  ngram_range=(3, 5), sublinear_tf=True)
    scaler     = StandardScaler()

    text_word = tfidf_word.fit_transform(df["clean_text"].fillna(""))
    text_char = tfidf_char.fit_transform(df["clean_text"].fillna(""))
    num       = sp.csr_matrix(scaler.fit_transform(df[num_cols].fillna(0).values))
    X         = sp.hstack([text_word, text_char, num], format="csr")

    # Best params from v3 tuning
    xgb_model = xgb.XGBClassifier(
        objective="multi:softmax", num_class=len(le.classes_),
        n_estimators=300, learning_rate=0.1, max_depth=8,
        subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
        reg_alpha=0.0, reg_lambda=0.5, tree_method="hist",
        random_state=42, eval_metric="mlogloss", verbosity=0,
    )
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(le.classes_),
        n_estimators=300, learning_rate=0.1, max_depth=8,
        subsample=0.9, colsample_bytree=0.8, class_weight="balanced",
        random_state=42, verbose=-1,
    )
    et_model = ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    model = VotingClassifier(
        estimators=[("xgb", xgb_model), ("lgb", lgb_model), ("et", et_model)],
        voting="soft", n_jobs=1,
    )

    print("Training model on full dataset...")
    model.fit(X, y)

    # Bundle everything needed for inference
    bundle = {
        "model":      model,
        "tfidf_word": tfidf_word,
        "tfidf_char": tfidf_char,
        "scaler":     scaler,
        "le":         le,
        "num_cols":   num_cols,
    }
    with open("model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("\nModel saved to model_bundle.pkl")
    print(f"Classes: {list(le.classes_)}")


# ============================================================
#  SHARED HELPER — load bundle + transform raw input
# ============================================================

def load_bundle(path="model_bundle.pkl"):
    if not os.path.exists(path):
        print("ERROR: model_bundle.pkl not found.")
        print("Run this first:  python test_model.py --mode save --data your_data.csv")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def transform(bundle, texts, num_array):
    """Apply saved transformers to raw inputs."""
    word = bundle["tfidf_word"].transform(texts)
    char = bundle["tfidf_char"].transform(texts)
    num  = sp.csr_matrix(bundle["scaler"].transform(num_array))
    return sp.hstack([word, char, num], format="csr")


# ============================================================
#  MODE A — Predict on a new CSV file
#  Run:  python test_model.py --mode csv --data "path/to/new_file.csv"
#  The CSV must have the same columns as the training data.
#  If it has a risk_label column, a full evaluation report is shown.
#  If not, predictions are just saved to predictions.csv.
# ============================================================

def test_on_csv(data_path):
    bundle = load_bundle()
    le     = bundle["le"]
    nc     = bundle["num_cols"]

    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    df["clean_text"] = df["clean_text"].fillna("")

    X = transform(bundle, df["clean_text"], df[nc].fillna(0).values)
    df["predicted_risk_label"] = le.inverse_transform(bundle["model"].predict(X))

    # Confidence scores (probability of the winning class)
    probs = bundle["model"].predict_proba(X)
    df["confidence"] = (np.max(probs, axis=1) * 100).round(1).astype(str) + "%"

    print("\n=== Sample Predictions (first 10) ===")
    cols = ["clause_id", "clean_text", "predicted_risk_label", "confidence"]
    if "risk_label" in df.columns:
        cols.insert(3, "risk_label")
    print(df.head(10)[cols].to_string(index=False))

    # If ground-truth labels exist → full evaluation
    if "risk_label" in df.columns:
        y_true = le.transform(df["risk_label"])
        y_pred = le.transform(df["predicted_risk_label"])
        critical_idx = list(le.classes_).index("Critical")

        print("\n=== Classification Report ===")
        print(classification_report(df["risk_label"], df["predicted_risk_label"], digits=4))

        per_class = recall_score(y_true, y_pred, average=None)
        print(f"Critical Clause Recall: {per_class[critical_idx]:.4f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

    out_path = "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved all predictions to {out_path}")


# ============================================================
#  MODE B — Interactive: type a clause, get a prediction
#  Run:  python test_model.py --mode interactive
#  Numeric features default to 0 — you can edit them in the prompt.
# ============================================================

def interactive_mode():
    bundle = load_bundle()
    le     = bundle["le"]
    nc     = bundle["num_cols"]

    severity_emoji = {"Critical": "CRITICAL", "High": "HIGH",
                      "Medium":   "MEDIUM",   "Low":  "LOW"}

    print("\n=== Interactive Clause Tester ===")
    print("Type a clause and press Enter. Type 'quit' to exit.\n")

    while True:
        clause = input("Clause text: ").strip()
        if clause.lower() in ("quit", "exit", "q"):
            break
        if not clause:
            continue

        # Optional: ask for numeric features
        print("Numeric features (press Enter to use 0 for all):")
        num_vals = []
        for col in nc:
            val = input(f"  {col} [0]: ").strip()
            try:
                num_vals.append(float(val) if val else 0.0)
            except ValueError:
                num_vals.append(0.0)

        X = transform(bundle, pd.Series([clause]),
                      np.array([num_vals]))

        pred  = bundle["model"].predict(X)[0]
        probs = bundle["model"].predict_proba(X)[0]
        label = le.inverse_transform([pred])[0]

        print(f"\n  Prediction  : [{severity_emoji[label]}] {label}")
        print("  Confidence breakdown:")
        for i, cls in enumerate(le.classes_):
            bar = "█" * int(probs[i] * 20)
            print(f"    {cls:<10s} {probs[i]*100:5.1f}%  {bar}")
        print()


# ============================================================
#  MODE C — Error analysis
#  Run:  python test_model.py --mode errors --data "path/to/data.csv"
#  Shows every clause the model got wrong, sorted by confidence.
#  Helps you understand failure patterns.
# ============================================================

def error_analysis(data_path):
    bundle = load_bundle()
    le     = bundle["le"]
    nc     = bundle["num_cols"]

    df = pd.read_csv(data_path)
    if "risk_label" not in df.columns:
        print("ERROR: CSV must have a 'risk_label' column for error analysis.")
        sys.exit(1)

    df["clean_text"] = df["clean_text"].fillna("")
    X = transform(bundle, df["clean_text"], df[nc].fillna(0).values)

    y_true = le.transform(df["risk_label"])
    y_pred = bundle["model"].predict(X)
    probs  = bundle["model"].predict_proba(X)

    df["predicted_risk_label"] = le.inverse_transform(y_pred)
    df["confidence"]           = np.max(probs, axis=1).round(3)
    df["correct"]              = df["risk_label"] == df["predicted_risk_label"]

    errors = df[~df["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False)

    total    = len(df)
    n_errors = len(errors)
    print(f"\nTotal clauses : {total}")
    print(f"Correct       : {total - n_errors}  ({(total-n_errors)/total*100:.1f}%)")
    print(f"Wrong         : {n_errors}  ({n_errors/total*100:.1f}%)")

    print("\n=== Most Confident Mistakes (model was very sure but wrong) ===")
    top_errors = errors.head(20)[
        ["clause_id", "clean_text", "risk_label",
         "predicted_risk_label", "confidence"]
    ]
    pd.set_option("display.max_colwidth", 60)
    print(top_errors.to_string(index=False))

    print("\n=== Confusion breakdown (what gets confused with what) ===")
    for true_cls in le.classes_:
        subset = errors[errors["risk_label"] == true_cls]
        if len(subset) == 0:
            continue
        counts = subset["predicted_risk_label"].value_counts()
        print(f"  True={true_cls:<10s}  predicted as: "
              + ",  ".join(f"{k}({v})" for k, v in counts.items()))

    errors.to_csv("errors.csv", index=False)
    print("\nAll errors saved to errors.csv")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the clause risk classifier")
    parser.add_argument("--mode", required=True,
                        choices=["save", "csv", "interactive", "errors"],
                        help="save | csv | interactive | errors")
    parser.add_argument("--data",
                        default=r"C:\Users\Satya\Downloads\master_dataset.csv",
                        help="Path to CSV (needed for save / csv / errors modes)")
    args = parser.parse_args()

    if args.mode == "save":
        data_path = args.data.strip().strip("\"'")
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {data_path}")
            sys.exit(1)
        save_model(data_path)

    elif args.mode == "csv":
        data_path = args.data.strip().strip("\"'")
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {data_path}")
            sys.exit(1)
        test_on_csv(data_path)

    elif args.mode == "interactive":
        interactive_mode()

    elif args.mode == "errors":
        data_path = args.data.strip().strip("\"'")
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {data_path}")
            sys.exit(1)
        error_analysis(data_path)
