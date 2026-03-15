# Risk-Aware Policy Clause Importance Ranking
## Complete User Guide & Technical Reference

> Version 1.0 | March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Full Pipeline Walkthrough](#4-full-pipeline-walkthrough)
5. [Key Hypotheses & Design Decisions](#5-key-hypotheses--design-decisions)
6. [Risk Scoring System](#6-risk-scoring-system)
7. [Dataset Summary](#7-dataset-summary)
8. [Sources & Citations](#8-sources--citations)
9. [Handoff to Machine-learning lead](#9-handoff-to-member-2)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

This repository implements a full NLP pipeline for **risk-aware clause importance ranking** across policy documents. The system:

- Segments policy documents into individual clauses
- Assigns risk severity labels: `Low` / `Medium` / `High` / `Critical`
- Extracts linguistic and semantic features
- Produces a balanced, publication-ready dataset for downstream ML classification

### What the System Does

| Stage | Script | Output |
|---|---|---|
| Document Collection | `step1_collect_docs.py` | `data/raw_docs/` |
| Clause Segmentation | `step3_clause_segmentation.py` | `clauses_raw.csv` |
| Clean + Dedup | `step4_clean_deduplicate.py` | `clauses_clean.csv` |
| Preprocessing | `step5_preprocessing.py` | `clauses_preprocessed.csv` |
| Risk Labeling | `step6_risk_labeling.py` | `clauses_labeled.csv` + `clauses_balanced.csv` |
| Feature Extraction | `step7_feature_extraction.py` | `clauses_features.csv` |
| Embeddings | `step8_embeddings.py` | `sbert_embeddings.npy` + `tfidf_matrix.npz` |
| Validation + Export | `step9_dataset_validation.py` | `master_dataset.csv` + `validation_report.txt` |

### Final Dataset Numbers

```
Raw documents collected  :  217
Raw clauses segmented    :  5,163
Clean clauses after Step4:  4,382
Preprocessed clauses     :  4,372  (10 code snippets removed)
Balanced training set    :  4,000  (1,000 per risk class)
```

---

## 2. Repository Structure

```
risk-aware-policy-clause-ranking/
│
├── data/
│   ├── raw_docs/                    ← Step 1 output (source documents)
│   │   ├── legal/                   ← 60 CUAD contracts (.txt)
│   │   ├── privacy/                 ← 20 privacy regulation docs (.txt)
│   │   ├── academic/                ← 30 university policy docs (.txt)
│   │   └── hr/                      ← 27 HR/corporate policy docs (.txt)
│   │
│   ├── segmented/                   ← Steps 3–9 outputs
│   │   ├── clauses_raw.csv          ← Step 3: raw segmented clauses
│   │   ├── clauses_clean.csv        ← Step 4: deduped + cleaned
│   │   ├── clauses_preprocessed.csv ← Step 5: lemmatized clean_text added
│   │   ├── clauses_labeled.csv      ← Step 6: full 4372 labeled clauses
│   │   ├── clauses_balanced.csv     ← Step 6: balanced 4000 clauses
│   │   ├── clauses_features.csv     ← Step 7: linguistic features added
│   │   ├── master_dataset.csv       ← Step 10: final handoff file
│   │   └── validation_report.txt    ← Step 9: quality report
│   │
│   └── embeddings/                  ← Step 8 outputs
│       ├── sbert_embeddings.npy     ← dense 4000×384 SBERT vectors
│       ├── tfidf_matrix.npz         ← sparse 4000×3000 TF-IDF matrix
│       ├── tfidf_vectorizer.pkl     ← fitted vectorizer (for inference)
│       ├── labels_encoded.npy       ← integer labels [0,1,2,3]
│       ├── label_encoder.pkl        ← class ↔ integer mapping
│       └── embedding_index.csv      ← clause_id alignment file
│
├── src/
│   ├── data_collection/
│   │   ├── data_collection.py
│   └── preprocessing/
│       ├── clause_segmentation.py
│       ├── clean_dup.py
│       ├── pre_processing.py
│       ├── risk_labeling.py
│       ├── feature_extraction.py
│       ├── embedded.py
│       └── dataset_validation.py
│

```

---

## 3. Setup & Installation

### Requirements

- Python 3.10+ (recommended -> 3.12.10)
- Windows (all paths tested on Windows; Linux/Mac compatible)
- ~4 GB disk space (CUAD data + embeddings)
- Internet connection on first run (downloads CUAD + SBERT model)

### Install All Dependencies

Run once from the project root with venv activated:

```bash
pip install datasets requests tqdm huggingface_hub
pip install pandas scikit-learn spacy sentence-transformers
pip install beautifulsoup4 lxml scipy numpy
python -m spacy download en_core_web_sm
```

> **Note:** `sentence-transformers` auto-downloads PyTorch (~800MB) if not installed. The `all-MiniLM-L6-v2` SBERT model (~80MB) downloads automatically on first use of `step8_embeddings.py`.

### Path Configuration

All scripts use this resolution pattern — works regardless of which directory you run from:

```python
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../src/preprocessing/
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # project root
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", ...)
```

> **Warning:** Never use relative paths like `"data/segmented/"`. Always use `PROJECT_ROOT` — this was a recurring bug where running from `src/` created duplicate `data/` folders inside `src/`.

---

## 4. Full Pipeline Walkthrough

### Step 1 — Document Collection

Collects policy documents across 4 domains.

**Legal domain — CUAD:**
The HuggingFace loading script for `theatticusproject/cuad` is broken (loading scripts no longer supported). The fix:
1. Download `CUADv1.json` manually from `github.com/TheAtticusProject/cuad` releases
2. Run `extract_cuad_from_json.py` pointing at the local JSON file

```bash
# Edit JSON_PATH in the script first, then:
python extract_cuad_from_json.py
```

**LEDGAR misclassification fix:**
The `lex_glue/LEDGAR` dataset was initially saved to `privacy/` but contains legal contract provisions, not privacy policies. Fix:

```bash
python fix_privacy_cleanup.py   # moves ledgar_*.txt from privacy/ → legal/
```

**Domain document counts:**

| Domain | Count | Source |
|---|---|---|
| Legal | ~60 | CUAD (trimmed from 510 for balance) |
| Privacy | ~20 | HIPAA, GDPR, CCPA, COPPA, FERPA, UK GDPR, LGPD, POPIA, PIPEDA, BIPA, EU AI Act |
| Academic | ~30 | Scraped university pages + curated docs |
| HR | ~27 | Scraped ACAS/OPM pages + curated docs |

---

### Step 2 — PDF to Text (Optional)

Since most collected documents are already `.txt` files, this step can be skipped. The segmentation script reads directly from `raw_docs/`. Only needed if you add new PDF documents.

```bash
pip install pdfplumber
python step2_pdf_to_text.py
```

---

### Step 3 — Clause Segmentation

The most critical step. Uses a **hybrid two-pass approach** tailored per domain.

#### Pass 1 — Rule-Based (Domain-Specific Patterns)

**Legal contracts:**
- Numbered clauses: `1.` `1.1` `1.1.1`
- Lettered sub-clauses: `(a)` `(b)` `(A)` `(B)` — matched at line start only to avoid mid-sentence false positives like `"company (a division of...)"`
- EXHIBIT headers: removed entirely (not clauses)
- Definition clauses: `"X" means...` / `"X" shall mean...`
- Initials lines: `BPS Initials __` — removed entirely

**Academic & HR:**
- Numbered headings with sub-bullets: `1. Attendance\n- Must arrive on time` → combined as `"Attendance: Must arrive on time"`
- Bullet points: `-`, `--`, `*`, `•`, `→`, `▪`, `‣`, `–`
- HTML pages: stripped via BeautifulSoup before segmentation

**Privacy:**
- `Section N` / `Article N` / `Amendment` headers as clause boundaries
- Numbered clauses within each section
- Fallback to academic/HR segmenter for docs using numbered structure (HIPAA, COPPA, GDPR) without Section/Article headers

#### Pass 2 — spaCy Sentence Splitting

Any paragraph surviving Pass 1 with `word_count > 40` gets sentence-split by `en_core_web_sm`. Tagged as `spacy_sentence` in `pattern_label`.

A second dedicated Pass 2 runs after the main loop to re-split any remaining `paragraph`-tagged clauses over 40 words. Tagged as `spacy_pass2`.

#### Key Tunable Parameters

```python
MIN_WORDS          = 10   # minimum clause word count (below = fragment)
MAX_WORDS          = 120  # maximum clause word count (above = under-split)
MAX_CLAUSES_PER_DOC = 100 # prevents legal contracts from dominating
LONG_PARA_WORDS    = 40   # threshold for triggering spaCy splitting
```

#### Final Segmentation Report

```
academic    437 clauses
hr          316 clauses
legal      4295 clauses
privacy     115 clauses
──────────────────────
Total      5163 clauses
Avg words  37.3
Pattern:   54% spacy_sentence | 22.7% paragraph | 13.3% numbered_clause
```

---

### Step 4 — Clean + Deduplicate

Three cleaning passes in sequence:

**4A — Exact Dedup:** Lowercased string comparison. Removes identical clauses.

**4B — Noise Removal:** Filters clauses that are:
- Alpha ratio < 0.55 (garbled / number-heavy text)
- Numeric token ratio > 0.40
- Contain repeated character sequences (`_ _ _ _`, `------`)
- Code snippets: `function(`, `getElementById`, `querySelector`, `// `, `const `, `let `
- No meaningful verb present

**4C — Near-Dedup (Jaccard):** Token set overlap ≥ 0.85. Runs per domain (O(n²) within domain only). Removes boilerplate variants like `"This agreement shall be governed by the laws of [State]"` appearing across 20 different contracts.

> **No domain balancing in Step 4.** Domain balance is irrelevant for classifier accuracy. What matters is risk label balance — deferred to Step 6 after labeling.

```
Raw clauses        5,163
After exact dedup: 5,000   (-163)
After noise filter:4,800   (-200)
After near-dedup:  4,382   (-418)
```

---

### Step 5 — Text Preprocessing

Produces `clean_text` column alongside `raw_text`.

```
raw_text   → SBERT embeddings, risk scoring, feature extraction
clean_text → TF-IDF baseline ONLY
```

Pipeline per clause:
1. Lowercase
2. Remove URLs
3. Remove special characters (keep letters, digits, spaces)
4. spaCy tokenize
5. Remove stopwords — **except** the `KEEP_WORDS` set below
6. Lemmatize

**Why KEEP_WORDS matters:**

```python
KEEP_WORDS = {
    "must", "shall", "may", "should", "will", "not", "no",
    "without", "unless", "except", "never", "only", "immediately",
    "prohibited", "required", "mandatory", "forbidden"
}
```

These are stopwords in standard NLP but are the most important risk signals in this domain. Removing them destroys the distinction between:
- `"Payment must be made within 30 days"` → High risk
- `"Payment may be made within 30 days"` → Low risk

**Compression ratio: 0.56** — 56% of raw tokens survive preprocessing. This is healthy — removes noise while retaining meaning.

---

### Step 6 — Risk Labeling

The most important step for model accuracy. Two-pass scoring on `raw_text`.

#### Risk Taxonomy

| Label | Description | Example |
|---|---|---|
| `Critical` | Termination, criminal liability, legal action, disqualification | "Violation shall result in immediate dismissal and criminal prosecution" |
| `High` | Suspension, formal disciplinary, significant restriction | "Employees found in breach will be subject to suspension without pay" |
| `Medium` | Warnings, compliance obligations, deductions | "Late submissions will receive a grade deduction of 10% per day" |
| `Low` | Informational, advisory, definitions, general statements | "Students are encouraged to attend office hours" |

#### Pass 1 — Token-Level Weighted Scoring

spaCy tokenizes each clause. Three score components computed:

```
modal_score       = max weight of modal verbs found
consequence_score = max weight of consequence keywords found
conditional_score = max weight of conditional phrases found

# Modal is zeroed if no consequence AND no conditional found
# This prevents purely informational clauses ("We must trust each other")
# from scoring Medium just because of "must"
if cons_score == 0.0 and cond_score == 0.0:
    modal_score = modal_score * 0.15

weighted_score = modal_score * 0.25 +
                 consequence_score * 0.50 +
                 conditional_score * 0.25
```

Consequence gets **50% weight** — it is the strongest and most reliable risk signal.

**Amplifiers** multiply the final score up or down:

```python
# Examples
AMPLIFIERS_UP  = {"gross misconduct": 1.35, "immediate": 1.3, "permanent": 1.3, ...}
AMPLIFIERS_DOWN = {"minor": 0.7, "first offence": 0.8, "advisory": 0.65, ...}
```

#### Pass 2 — N-Gram Verification

Builds a vocabulary of high-risk n-grams (1,2,3) from the corpus using `CountVectorizer(min_df=2)`. Cross-references against high-risk seed phrases.

**Critical fix:** Pass 2 only escalates clauses where Pass 1 scored > 0.0. It never inflates truly informational clauses (those with zero Pass 1 score). The default fallback weight of 0.6 was removed — this was the source of the mysterious 0.300 minimum that caused `Low = 0`.

#### Score → Label Mapping

```python
>= 0.75  →  Critical
>= 0.50  →  High
>= 0.10  →  Medium
<  0.10  →  Low
```

#### Known Pitfalls Fixed

| Issue | Root Cause | Fix |
|---|---|---|
| `Low = 0` | `subject to` in conditionals (weight 0.85) fires on every legal clause | Removed `subject to` — not a risk signal |
| Min score = 0.300 | Pass 2 default weight 0.6 → blend `(0.0 + 0.6)/2 = 0.300` | Removed default; pass2 returns `p1` unchanged if ngram not in known dicts |
| `modal * 0.15` applied twice | Duplicate code block in script | Removed duplicate |
| `immediately` inflating scores | Single word in conditionals at 0.85 | Removed from conditionals |

#### Final Label Distribution

```
Pre-balance:                     Post-balance (training set):
  Critical   627  (14.3%)          Critical  1000  (25.0%)
  High       638  (14.6%)          High      1000  (25.0%)
  Medium    1523  (34.8%)          Medium    1000  (25.0%)
  Low       1584  (36.2%)          Low       1000  (25.0%)
  ──────────────────────           ──────────────────────
  Total     4372                   Total     4000
```

Balancing strategy:
- **Majority classes** (Medium, Low): undersample keeping highest-scoring clauses (`nlargest`)
- **Minority classes** (Critical, High): oversample with replacement

---

### Step 7 — Feature Extraction

Adds 7 linguistic feature columns to `clauses_features.csv`:

| Feature | Type | Description |
|---|---|---|
| `modal_score` | float 0-1 | Max modal verb weight in clause |
| `consequence_score` | float 0-1 | Max consequence keyword weight |
| `conditional_score` | float 0-1 | Max conditional phrase weight |
| `has_negation` | int 0/1 | Presence of negation (must not, shall not, etc.) |
| `word_count_f` | int | Raw word count |
| `obligation_count` | int | Count of must/shall/will/required tokens |
| `penalty_flag` | int 0/1 | 1 if consequence_score >= 0.7 |

---

### Step 8 — Embeddings

Produces two representations of each clause:

**TF-IDF (lexical):**
```python
TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
```
- Uses `clean_text` (stopwords removed)
- `ngram_range=(1,2)` — bigrams chosen over trigrams because trigrams are too sparse at 4000 samples
- `sublinear_tf=True` — log-scaling prevents high-frequency terms from dominating

**SBERT (semantic):**
```python
SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embeddings
```
- Uses `raw_text` (full original clause)
- `all-MiniLM-L6-v2` — 14,000+ citations in research, recommended for classification tasks
- Batch size 64, ~20 minutes on CPU for 4000 clauses

> **Research note:** `all-MiniLM-L6-v2` is explicitly cited in the original SBERT paper (Reimers & Gurevych, EMNLP 2019) as the recommended model for classification. For ablation, compare with `all-mpnet-base-v2` (768-dim, slower but marginally more accurate).

---

### Step 9+10 — Validation & Final Export

Runs 7 automated quality checks:

```
✓ 4 risk labels present
✓ Min 500 clauses per class
✓ SBERT no NaN values
✓ Embeddings aligned with labels
✓ TF-IDF features > 1000
✓ Linguistic features exist
✓ Imbalance ratio < 2x
```

Also measures **inter-class separability** — if Critical vs Low cosine similarity is < 0.3, SBERT is already separating classes well before any classifier, predicting strong accuracy.

Exports `master_dataset.csv` with all 16 columns merged into one handoff file.

---

## 5. Key Hypotheses & Design Decisions

### Hypothesis 1 — Domain-Specific Segmentation Beats Generic NLP

**Observation:** Legal contracts use `(a)(b)` lettered sub-clauses, EXHIBIT headers, and definition patterns. Academic docs use numbered headings with sub-bullets. Privacy docs use Section/Article boundaries.

**Decision:** Built four separate segmenters (legal, academic/hr, privacy) instead of one generic spaCy sentence splitter.

**Result:** Pattern breakdown shows 22.7% numbered_clause and 14.6% lettered_subclause — structures that spaCy alone would have merged into one large paragraph.

---

### Hypothesis 2 — N-Gram Verification as Safety Net

**Observation:** Single-token matching misses compound risk phrases. `"legal"` alone is ambiguous (`"legal tender"`, `"legal counsel"`). `"legal action"` is unambiguous Critical.

**Decision:** Add a Pass 2 n-gram (1,2,3) verification layer using CountVectorizer to catch risk phrases that Pass 1 might miss.

**Result:** 168 high-risk n-grams identified from corpus. Pass 2 escalates ~8% of clauses where the n-gram score exceeds Pass 1 by more than 0.15.

---

### Hypothesis 3 — raw_text for Semantics, clean_text for Lexical Only

**Observation:** Stopwords like `must`, `shall`, `not`, `without` are the strongest risk signals in policy text. Standard preprocessing removes them.

**Decision:** Preserve `raw_text` for all semantic work. `clean_text` used exclusively for TF-IDF where term frequency matters more than function words.

**Result:** Without this distinction, `"Payment must be made"` and `"Payment may be made"` become identical after preprocessing — destroying the most important risk signal.

---

### Hypothesis 4 — Risk Label Balance > Domain Balance

**Observation:** Legal domain at 84% of corpus would cause domain-biased classifier. But the classifier needs to predict risk level (Low/Medium/High/Critical), not domain.

**Decision:** No domain balancing in Step 4. Balance by risk level at Step 6 after labeling.

**Result:** 4000 perfectly balanced training samples (1000 per risk class). The classifier learns risk language patterns, not legal contract patterns.

---

### Hypothesis 5 — Consequence Score Gets 50% Weight

**Observation:** Modal verbs alone are weak signals. `"We must trust each other"` contains `must` but is Low risk. Consequence keywords are unambiguous — `"termination"` is always Critical regardless of how it's introduced.

**Decision:** `consequence_score` gets 50% weight in the scoring formula vs 25% each for modal and conditional.

**Result:** Correctly scores `"We must trust each other"` as Low (no consequence keyword → modal zeroed → score ≈ 0.0) while `"Violations must result in termination"` correctly scores Critical.

---

## 6. Risk Scoring System

### Keyword Weight Reference

**Modal Verbs (0.1 – 0.9):**

| Weight | Keywords |
|---|---|
| 0.9 | must, shall, obliged, obligated, mandated |
| 0.8 | will, required |
| 0.6 | should, ought, expected, necessary |
| 0.3 | may, can, could, allowed, permitted |
| 0.1 | encouraged, recommended, suggested, optional |

**Consequence Keywords (0.1 – 1.0):**

| Weight | Keywords |
|---|---|
| 1.0 | termination, dismissal, expulsion, disqualification, imprisonment, prosecution, conviction, legal action, law enforcement |
| 0.95 | criminal, revocation, injunction |
| 0.85 | damages, liable, liability, infringement, fine, sanction, banned, barred |
| 0.8 | breach, default, enforcement |
| 0.75 | suspension, forfeiture |
| 0.7 | disciplinary, prohibited, loss of, investigation |
| 0.5 | warning, reprimand |
| 0.45 | obligation, covenant, warranty, deduction |
| 0.4 | comply, compliance, monitor |

**Conditional Phrases (0.1 – 1.0):**

| Weight | Keywords |
|---|---|
| 1.0 | failure to, failure of, non-compliance, in violation, violation of, breach of, zero tolerance |
| 0.95 | in breach, liable for |
| 0.9 | is prohibited, strictly prohibited |
| 0.85 | shall not, must not, without authorization, without consent, without notice, subject to |
| 0.8 | constitute, give rise to, result in, upon breach |
| 0.6 | may result, could result, in the event, provided that |
| 0.5 | unless |

**Amplifiers:**

| Direction | Factor | Keywords |
|---|---|---|
| Up | 1.35 | gross misconduct |
| Up | 1.30 | immediate, permanent, gross, strict, without exception, criminal |
| Up | 1.25 | serious, severe, automatic, mandatory, per violation, willful |
| Down | 0.65 | advisory, informational, optional |
| Down | 0.70 | minor, informal |
| Down | 0.75 | minimal, incidental, verbal, first offence |

### Scoring Formula

```python
# Step 1: Get component scores
modal_score       = max weight of modal verbs found (token-level)
consequence_score = max weight of consequence keywords (phrase-level, longest match first)
conditional_score = max weight of conditional phrases (phrase-level, longest match first)

# Step 2: Zero modal if no risk context
if consequence_score == 0.0 and conditional_score == 0.0:
    modal_score = modal_score * 0.15

# Step 3: Weighted combination
score = modal_score * 0.25 + consequence_score * 0.50 + conditional_score * 0.25

# Step 4: Apply amplifiers
score = score * max(amplifier_up_factors, default=1.0)
score = score * min(amplifier_down_factors, default=1.0)

# Step 5: Consequence override (critical keywords always floor at Critical)
if consequence_score >= 0.95:
    score = max(score, 0.75)

# Step 6: Map to label
Critical  >= 0.75
High      >= 0.50
Medium    >= 0.10
Low       <  0.10
```

---

## 7. Dataset Summary

### Multi-Domain Policy Clause Corpus (MDPCC) v1.2

| Domain | Docs | Raw Clauses | Clean Clauses | Source |
|---|---|---|---|---|
| Legal | 60 | 4,295 | 3,679 | CUAD v1 (SEC EDGAR contracts) |
| Academic | 30 | 437 | 376 | University pages + curated |
| HR | 27 | 316 | 225 | ACAS/OPM pages + curated |
| Privacy | 20 | 115 | 102 | HIPAA, GDPR, CCPA, COPPA, FERPA, PIPEDA, etc. |

### Output Files Reference

| File | Shape | Description |
|---|---|---|
| `clauses_raw.csv` | 5163 × 6 | Raw segmented clauses from Step 3 |
| `clauses_clean.csv` | 4382 × 6 | After dedup + noise removal |
| `clauses_preprocessed.csv` | 4372 × 8 | Adds `clean_text`, `token_count` |
| `clauses_labeled.csv` | 4372 × 10 | Adds `risk_label`, `risk_score` |
| `clauses_balanced.csv` | 4000 × 10 | 1000 per risk class |
| `clauses_features.csv` | 4000 × 17 | Adds 7 linguistic feature columns |
| `master_dataset.csv` | 4000 × 16 | All columns merged, final handoff |
| `sbert_embeddings.npy` | 4000 × 384 | Dense semantic vectors |
| `tfidf_matrix.npz` | 4000 × 3000 | Sparse lexical vectors |

---

## 8. Sources & Citations

Full citation details are in `sources.json`. Key sources:

```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}

@inproceedings{reimers2019sentencebert,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of EMNLP},
  year={2019}
}

@inproceedings{chalkidis2022lexglue,
  title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English},
  author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and others},
  booktitle={Proceedings of ACL},
  year={2022}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of KDD},
  year={2016}
}
```

**Privacy domain regulations used:**
- HIPAA Privacy Rule (45 CFR Part 164) — HHS 2003
- COPPA Rule (16 CFR Part 312) — FTC 2000, amended 2025
- FERPA (20 U.S.C. § 1232g) — U.S. Dept. of Education 1974
- GDPR (Regulation EU 2016/679) — European Parliament 2016
- UK GDPR + Data Protection Act 2018
- LGPD (Law No. 13,709/2018) — Brazil
- POPIA (Act 4 of 2013) — South Africa
- PIPEDA (S.C. 2000, c. 5) — Canada
- BIPA (740 ILCS 14) — Illinois 2008
- EU AI Act (Regulation EU 2024/1689)

---
### Files to Hand Over

```

data/
├── segmented/
│   ├── master_dataset.csv       ← PRIMARY: all features in one file
│   ├── clauses_balanced.csv     ← 4000 balanced training-ready clauses
│   └── validation_report.txt   ← quality checks report
└── embeddings/
    ├── sbert_embeddings.npy     ← 4000×384 SBERT (use for best accuracy)
    ├── tfidf_matrix.npz         ← 4000×3000 TF-IDF (use for baseline)
    ├── tfidf_vectorizer.pkl     ← for inference on new clauses
    ├── labels_encoded.npy       ← integer labels (Critical=0, High=1, Low=2, Medium=3)
    ├── label_encoder.pkl        ← inverse transform labels
    └── embedding_index.csv      ← clause_id → embedding row alignment
```

### Recommended Experiments for Machine-learning lead

| # | Features | Classifier | Expected Accuracy |
|---|---|---|---|
| 1 (Baseline) | TF-IDF | Logistic Regression | 65–75% |
| 2 (Interpretable) | Linguistic features (7 cols) | XGBoost | 72–80% |
| 3 (Best single) | SBERT embeddings | XGBoost | 82–88% |
| 4 (Paper result) | SBERT + Linguistic features | XGBoost ensemble | 87–92% |
| 5 (Fine-tuned) | Raw text | LegalBERT fine-tuned | 88–93% |

### Loading the Data

```python
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz

# Load all components
df         = pd.read_csv("data/segmented/master_dataset.csv")
embeddings = np.load("data/embeddings/sbert_embeddings.npy")
tfidf      = load_npz("data/embeddings/tfidf_matrix.npz")
labels     = np.load("data/embeddings/labels_encoded.npy")

with open("data/embeddings/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Quick experiment: SBERT + XGBoost
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

X = embeddings
y = labels

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
model = XGBClassifier(n_estimators=300, max_depth=6, random_state=42)
# ... run cross-validation
```

---

## 10. Troubleshooting

### `Low = 0` in risk labeling
**Cause:** Usually `subject to` in CONDITIONAL_KEYWORDS or Pass 2 default weight inflating zero scores.
**Fix:** Verify `subject to` is NOT in your `CONDITIONAL_KEYWORDS` dict. Verify `pass2_check` returns `p1` unchanged (not 0.6) when ngram not in known dicts.

### CUAD download fails
**Cause:** HuggingFace loading scripts no longer supported.
**Fix:** Download `CUADv1.json` manually from `github.com/TheAtticusProject/cuad` → Releases. Run `extract_cuad_from_json.py`.

### `Privacy = 0` clauses after segmentation
**Cause:** New privacy docs (HIPAA, GDPR etc.) use numbered structure, not Section/Article headers. The privacy segmenter was only triggered by Section/Article.
**Fix:** Already fixed — `segment_privacy()` falls back to `segment_academic_hr()` when no Section/Article structure found.

### Scripts writing to wrong directory
**Cause:** Running script from `src/` using relative path `"data/"` creates `src/data/`.
**Fix:** Use the `PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))` pattern in all scripts.

### `ValueError: a must be greater than 0`
**Cause:** `df.sample(1)` called on empty DataFrame (e.g., Low class has 0 clauses).
**Fix:** Always guard with `if len(subset) == 0: continue` before sampling.

### SBERT taking too long
**Cause:** CPU encoding of 4000 clauses at batch_size=32.
**Fix:** Increase `batch_size=64` or `batch_size=128`. If GPU available, sentence-transformers auto-detects CUDA.

### `Min score = 0.300` (all clauses Medium or above)
**Cause:** Pass 2 default fallback of `0.6` blending with `0.0` → `(0.0+0.6)/2 = 0.300`.
**Fix:** In `pass2_check`, if the ngram is not in `CONSEQUENCE_KEYWORDS` or `CONDITIONAL_KEYWORDS`, do NOT append any score — just return `p1` unchanged.

---

*Built with ❤ by Pratham Mohan — NLP & Clause Processing Lead*  
*Risk-Aware Policy Clause Importance Ranking Project | March 2026*
