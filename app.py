import os, re, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pdfplumber

from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "risk-ranker-secret"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Feature extraction keywords ────────────────────────────────────────────────
MODALS = {
    "must": 0.9, "shall": 0.9, "will": 0.8, "required": 0.8,
    "should": 0.6, "may": 0.3, "can": 0.3, "encouraged": 0.1, "optional": 0.1,
}
CONSEQUENCES = {
    "termination": 1.0, "dismissal": 1.0, "criminal": 0.95, "imprisonment": 1.0,
    "prosecution": 1.0, "damages": 0.85, "penalty": 0.85, "fine": 0.85,
    "suspension": 0.75, "prohibited": 0.7, "disciplinary": 0.7,
    "breach": 0.8, "liable": 0.85, "liability": 0.85, "sanction": 0.85,
    "warning": 0.5, "deduction": 0.45, "comply": 0.4, "obligation": 0.45,
}
CONDITIONALS = {
    "failure to": 1.0, "violation of": 1.0, "breach of": 1.0,
    "result in": 0.85, "constitute": 0.8, "shall not": 0.85,
    "must not": 0.85, "in the event of": 0.85, "without consent": 0.85,
    "may result": 0.6, "provided that": 0.6, "unless": 0.5,
}
NEGATIONS = [
    "must not", "shall not", "not permitted", "not allowed",
    "without consent", "without authorization", "strictly prohibited",
]

def extract_features(text):
    t = text.lower()
    tok = t.split()
    modal = max((MODALS[w] for w in tok if w in MODALS), default=0.0)
    cons = 0.0
    for p, w in sorted(CONSEQUENCES.items(), key=lambda x: -len(x[0])):
        if p in t: cons = max(cons, w); break
    cond = 0.0
    for p, w in sorted(CONDITIONALS.items(), key=lambda x: -len(x[0])):
        if p in t: cond = max(cond, w); break
    neg   = int(any(n in t for n in NEGATIONS))
    wc    = len(text.split())
    oblig = sum(1 for w in tok if w in {"must", "shall", "will", "required"})
    pflag = int(cons >= 0.7)
    return modal, cons, cond, neg, oblig, pflag, wc

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ── Clause segmentation ────────────────────────────────────────────────────────
NUMBERED  = re.compile(r"(?:^|\n)\s*(\d{1,2}[\.\)]\s+|\(?[a-z]\)\s+|[ivxlIVXL]+\.\s+)")
SEMICOLON = re.compile(r";\s+")

def segment_clauses(text):
    paragraphs = re.split(r"\n{2,}", text)
    clauses = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = NUMBERED.split(para)
        chunks = [p.strip() for p in parts if p and not re.fullmatch(r"[\d\.\)\(\s]+", p)]
        if len(chunks) <= 1:
            chunks = [c.strip() for c in SEMICOLON.split(para)]
        for chunk in chunks:
            chunk = re.sub(r"\s+", " ", chunk).strip()
            wc = len(chunk.split())
            if 8 <= wc <= 150:
                clauses.append(chunk)
            elif wc > 150:
                for s in re.split(r"(?<=[.!?])\s+", chunk):
                    s = s.strip()
                    if 8 <= len(s.split()) <= 150:
                        clauses.append(s)
    return clauses

# ── Model training ─────────────────────────────────────────────────────────────
NUM_COLS = [
    "modal_score", "consequence_score", "conditional_score",
    "has_negation", "obligation_count", "penalty_flag", "word_count",
]


MODEL_PKL = os.path.join(os.path.dirname(__file__), "src", "Machine_Learning", "RiskEnsembleClassifier.pkl")

BUNDLE = None

def get_bundle():
    global BUNDLE
    if BUNDLE is not None:
        return BUNDLE
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PKL}\n"
            "Run this first:\n"
            "  cd src/Machine_Learning\n"
            "  python test_model.py --mode save --data ../../output/master_dataset.csv"
        )
    print("[RiskApp] Loading model...")
    with open(MODEL_PKL, "rb") as f:
        BUNDLE = pickle.load(f)
    return BUNDLE

# ── Inference ──────────────────────────────────────────────────────────────────
SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
BADGE_COLOR    = {"Low": "low", "Medium": "medium", "High": "high", "Critical": "critical"}

def predict_clauses(raw_clauses):
    bundle = get_bundle()
    le = bundle["label_encoder"]  
    model = bundle["model"]

    clean_texts = [preprocess_text(c) for c in raw_clauses]
    feat_rows   = [extract_features(c) for c in raw_clauses]
    num_array   = np.array(feat_rows, dtype=float)

    word = bundle["tfidf_word"].transform(clean_texts)
    char = bundle["tfidf_char"].transform(clean_texts)
    num  = sp.csr_matrix(bundle["scaler"].transform(num_array))
    X    = sp.hstack([word, char, num], format="csr")

    preds = le.inverse_transform(model.predict(X))
    probs = model.predict_proba(X)
    confs = (np.max(probs, axis=1) * 100).round(1)

    results = []
    for i, clause in enumerate(raw_clauses):
        label = preds[i]
        results.append({
            "clause": clause, "label": label,
            "confidence": float(confs[i]),
            "badge": BADGE_COLOR.get(label, "low"),
            "severity": SEVERITY_ORDER.get(label, 1),
        })

    results.sort(key=lambda x: (-x["severity"], -x["confidence"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = request.form.get("input_text")
    file = request.files.get("pdf_file")

    text = ""

    if text_input and text_input.strip():
        text = text_input
        filename = "Manual Input"

    elif file and file.filename != "":
        if not file.filename.lower().endswith(".pdf"):
            flash("Only PDF files allowed.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            flash(f"Failed to read PDF: {e}")
            return redirect(url_for("index"))
        finally:
            try:
                os.remove(filepath)
            except:
                pass

    else:
        flash("Please upload a PDF or paste text.")
        return redirect(url_for("index"))

    if not text.strip():
        flash("No readable content found.")
        return redirect(url_for("index"))

    clauses = segment_clauses(text)

    if not clauses:
        flash("No clauses could be extracted.")
        return redirect(url_for("index"))

    results = predict_clauses(clauses)

    summary = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for r in results:
        summary[r["label"]] = summary.get(r["label"], 0) + 1

    return render_template(
        "results.html",
        filename=filename,
        results=results,
        summary=summary,
        total=len(results)
    )
    
if __name__ == "__main__":
    print("[RiskApp] Pre-loading model...")
    get_bundle()
    app.run(debug=False, host="0.0.0.0", port=7860)