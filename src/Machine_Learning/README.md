# RiskEnsembleClassifier
### Risk-Aware Policy Clause Importance Ranking — Member 2: Risk Modeling & ML Lead

---

## What This Does

Takes a policy document's clauses and automatically predicts the **risk severity** of each clause (Low / Medium / High / Critical), then ranks them by risk so the most dangerous clauses appear first.

---

## Folder Structure

```
src/
└── Machine_Learning/
    ├── RiskEnsembleClassifier.py   ← main training + evaluation script
    ├── test_model.py               ← testing script (csv / interactive / error analysis)
    ├── RiskEnsembleClassifier.pkl  ← saved model bundle (gitignored if large)
    ├── ranked_clauses.csv          ← output: all clauses ranked by risk score
    └── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate the model
python src/Machine_Learning/RiskEnsembleClassifier.py

# Or with custom data path
python src/Machine_Learning/RiskEnsembleClassifier.py --data "path/to/dataset.csv"
```

---

## Models Trained

Three baseline models were trained and compared before arriving at the final ensemble.

### Model 1 — Logistic Regression (Baseline)

Simple linear classifier used as the starting reference point.

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Critical | 0.7823    | 0.6950 | 0.7361   | 200     |
| High     | 0.6912    | 0.7150 | 0.7029   | 200     |
| Low      | 0.7234    | 0.8100 | 0.7644   | 200     |
| Medium   | 0.6978    | 0.6700 | 0.6836   | 200     |
| **Overall Accuracy** | | | **0.7225** | **800** |

**Critical Clause Recall: 0.6950**

---

### Model 2 — Random Forest

Ensemble of decision trees, better than logistic regression but still limited by bag-of-words features.

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Critical | 0.8534    | 0.8050 | 0.8285   | 200     |
| High     | 0.7823    | 0.8000 | 0.7910   | 200     |
| Low      | 0.7945    | 0.8750 | 0.8329   | 200     |
| Medium   | 0.7812    | 0.7450 | 0.7627   | 200     |
| **Overall Accuracy** | | | **0.8063** | **800** |

**Critical Clause Recall: 0.8050**

---

### Model 3 — XGBoost (Tuned)

Gradient boosting with hyperparameter tuning via RandomizedSearchCV.

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Critical | 1.0000    | 0.9050 | 0.9501   | 200     |
| High     | 0.8955    | 0.9000 | 0.8978   | 200     |
| Low      | 0.8609    | 0.9900 | 0.9209   | 200     |
| Medium   | 0.8936    | 0.8400 | 0.8660   | 200     |
| **Overall Accuracy** | | | **0.9088** | **800** |

**Critical Clause Recall: 0.9050**

---

### ✅ Final Model — RiskEnsembleClassifier (XGBoost + LightGBM + ExtraTrees)

Best performing model. Combines 3 models using soft voting (averages class probabilities).

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Critical | 1.0000    | 0.9000 | 0.9474   | 200     |
| High     | 0.9069    | 0.9250 | 0.9158   | 200     |
| Low      | 0.8795    | 0.9850 | 0.9292   | 200     |
| Medium   | 0.8854    | 0.8500 | 0.8673   | 200     |
| **Overall Accuracy** | | | **0.9150** | **800** |

**Critical Clause Recall: 0.9000**

---

## Model Comparison Summary

| Model | Accuracy | Critical Recall | Critical F1 | Notes |
|-------|----------|-----------------|-------------|-------|
| Logistic Regression | 72.25% | 0.695 | 0.736 | Baseline |
| Random Forest | 80.63% | 0.805 | 0.829 | Better than baseline |
| XGBoost (Tuned) | 90.88% | 0.905 | 0.950 | Big jump with tuning |
| **RiskEnsembleClassifier** | **91.50%** | **0.900** | **0.947** | **Final model** |

> **Why XGBoost alone scored higher Critical Recall (0.905) vs Ensemble (0.900)?**
> The ensemble slightly smooths out extreme predictions — it trades a tiny bit of critical recall for better overall balance across all 4 classes. The ensemble wins on overall accuracy (91.5% vs 90.88%).

---

## Confusion Matrix (Final Model)

```
              Predicted
              Critical  High  Low  Medium
Actual  Critical   180    11    0       9
        High         0   185    5      10
        Low          0     0  197       3
        Medium       0     8   22     170
```

Key observations:
- Critical → only 9 misclassified as Medium, 11 as High. Zero misclassified as Low ✓
- Low → nearly perfect (197/200 correct)
- Medium → 22 confused with Low (hardest class to distinguish)

---

## Feature Engineering

| Feature Type | Method | Count |
|---|---|---|
| Word n-grams | TF-IDF (1,2)-grams, sublinear_tf | 2000 |
| Char n-grams | TF-IDF (3,5)-grams, char_wb | 1000 |
| Numeric features | modal_score, consequence_score, conditional_score, has_negation, obligation_count, penalty_flag, word_count | 7 |
| **Total features** | | **3007** |

### Top 15 Important Features (XGBoost sub-model)

| Feature | Importance |
|---|---|
| ment (char) | 0.0215 |
| rmati (char) | 0.0143 |
| penalty_flag | 0.0123 |
| men (char) | 0.0112 |
| eri (char) | 0.0112 |
| acti (char) | 0.0101 |
| section agreement | 0.0085 |
| shall conduct | 0.0075 |
| consequence_score | 0.0069 |
| has_negation | 0.0065 |

---

## Difficulties Faced & How Overcome

### 1. Data Leakage in TF-IDF
**Problem:** Initially `tfidf.fit_transform()` was called on the full dataset before the train/test split. This meant the model had seen test-set vocabulary during training, giving artificially inflated metrics.

**Solution:** Restructured the pipeline — `fit_transform` only on training data, `transform` on test data. Accuracy dropped slightly but became trustworthy.

---

### 2. Memory Issue with Dense Matrices
**Problem:** Calling `.toarray()` on TF-IDF output converted sparse matrices to dense numpy arrays. With 3000+ features × 4000 rows this consumed ~100MB+ RAM and slowed everything down.

**Solution:** Kept all matrices in sparse format (`scipy.sparse.csr_matrix`) throughout and used `sp.hstack()` for combining. Memory usage dropped by ~10x.

---

### 3. GridSearchCV Taking Too Long
**Problem:** Original `GridSearchCV` with 4×3×3×3 = 108 combinations × 3 folds = 324 full XGBoost training runs. On this dataset it would have taken hours.

**Solution:** Switched to `RandomizedSearchCV` with `n_iter=20` — samples 20 random combinations instead of all 108. Runs in ~10 minutes with near-equivalent results.

---

### 4. Wrong Scoring Metric in Hyperparameter Search
**Problem:** GridSearchCV was using `scoring='accuracy'` but the most important metric for this project is **Critical clause recall** — a model that ignores the Critical class can still score high accuracy.

**Solution:** Created a custom scorer combining `f1_macro` + `critical_recall` (50/50 weight). This ensures hyperparameter search optimises for what actually matters.

---

### 5. Class Imbalance Handling
**Problem:** `sample_weight` was passed to `RandomizedSearchCV` causing a `UserWarning` about statistically incorrect results since the custom scorer doesn't support sample weights.

**Solution:** Moved class balancing inside each model using `class_weight="balanced"` parameter (LightGBM, ExtraTrees) instead of external sample weights. Warning eliminated.

---

### 6. IndentationError Crashing Script
**Problem:** A mid-file `import` statement had 4 spaces of accidental indentation, causing an `IndentationError` before the GridSearch even ran.

**Solution:** Moved all imports to the top of the file following PEP8 conventions.

---

## Saved Model

The trained model is exported as a pickle bundle containing everything needed for inference:

```python
import pickle, scipy.sparse as sp

with open("RiskEnsembleClassifier.pkl", "rb") as f:
    bundle = pickle.load(f)

# Predict on new clause
text  = ["employee must disclose all conflicts of interest immediately"]
word  = bundle["tfidf_word"].transform(text)
char  = bundle["tfidf_char"].transform(text)
num   = sp.csr_matrix(bundle["scaler"].transform([[0]*7]))
X_new = sp.hstack([word, char, num], format="csr")

pred  = bundle["model"].predict(X_new)
label = bundle["label_encoder"].inverse_transform(pred)
print(label)  # ['Critical']
```

---

## Tech Stack

```
Python 3.13
scikit-learn    — TF-IDF, LabelEncoder, VotingClassifier, ExtraTrees, metrics
xgboost         — XGBClassifier
lightgbm        — LGBMClassifier
scipy           — sparse matrix operations
pandas          — data handling
numpy           — numerical ops
pickle          — model serialisation
```

---

## Output Files

| File | Description |
|---|---|
| `RiskEnsembleClassifier.pkl` | Saved model bundle (model + transformers + encoder) |
| `ranked_clauses.csv` | All 4000 clauses ranked by composite risk score |
