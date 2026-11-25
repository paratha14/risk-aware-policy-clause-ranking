import io, json, os, re, pickle, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from time import perf_counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pdfplumber

from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from llm_segmentor import segment_clauses

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

# ── Model loading ──────────────────────────────────────────────────────────────
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


def warm_up_model():
    """Load the serialized inference bundle before serving user requests."""
    start_time = perf_counter()
    get_bundle()
    print(f"[RiskApp] Model ready in {perf_counter() - start_time:.2f}s")


# Gunicorn imports this module instead of executing the __main__ block.  Loading
# here avoids a slow first /analyze request; with Gunicorn --preload the loaded
# bundle is also shared with worker processes through copy-on-write memory.
warm_up_model()

# ── Inference ──────────────────────────────────────────────────────────────────
SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
BADGE_COLOR    = {"Low": "low", "Medium": "medium", "High": "high", "Critical": "critical"}
PDF_RISK_COLORS = {
    "Critical": colors.HexColor("#FECACA"), "High": colors.HexColor("#FED7AA"),
    "Medium": colors.HexColor("#FEF3C7"), "Low": colors.HexColor("#DCFCE7"),
}
PDF_RISK_TEXT_COLORS = {
    "Critical": colors.HexColor("#991B1B"), "High": colors.HexColor("#9A3412"),
    "Medium": colors.HexColor("#854D0E"), "Low": colors.HexColor("#166534"),
}

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


def _pdf_safe(value):
    """Keep the report compatible with ReportLab's built-in Helvetica font."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def build_risk_report(results, filename):
    """Create the downloadable report in memory; no report is written to disk."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RiskReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "RiskReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"), spaceAfter=8,
    )
    clause_style = ParagraphStyle(
        "RiskClause", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
        leading=13, textColor=colors.HexColor("#1E293B"),
    )
    label_style = ParagraphStyle(
        "RiskLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8,
        leading=11,
    )

    story = [
        Paragraph("Policy Clause Risk Report", title_style),
        Paragraph(f"Source: {escape(_pdf_safe(filename))} &middot; {len(results)} clauses analysed", subtitle_style),
        Spacer(1, 4 * mm),
    ]
    for result in results:
        label = result.get("label", "Low")
        color = PDF_RISK_COLORS.get(label, colors.white)
        label_color = PDF_RISK_TEXT_COLORS.get(label, colors.HexColor("#1E293B"))
        header = Paragraph(
            f"#{result.get('rank', '')}  {escape(_pdf_safe(label.upper()))}"
            f"  |  Confidence: {float(result.get('confidence', 0)):.1f}%",
            ParagraphStyle("RiskLabelColor", parent=label_style, textColor=label_color),
        )
        clause = Paragraph(escape(_pdf_safe(result.get("clause", ""))), clause_style)
        card = Table([[header], [clause]], colWidths=[178 * mm])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("BOX", (0, 0), (-1, -1), 0.7, label_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 7), ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
        ]))
        story.extend([card, Spacer(1, 4 * mm)])
    document.build(story)
    buffer.seek(0)
    return buffer

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/download-report", methods=["POST"])
def download_report():
    try:
        results = json.loads(request.form["results_json"])
        if not isinstance(results, list):
            raise ValueError("Results must be a list")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        flash("The report data was invalid. Please analyse the document again.")
        return redirect(url_for("index"))

    source_name = request.form.get("filename", "policy-document")
    report_name = secure_filename(os.path.splitext(source_name)[0]) or "policy-document"
    return send_file(
        build_risk_report(results, source_name), mimetype="application/pdf",
        as_attachment=True, download_name=f"{report_name}-risk-report.pdf",
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = datetime.now()
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
    file_save= datetime.now()
    print("time taken to save and extract the text:", file_save - start_time)
    try:
        clauses = segment_clauses(text)
        clauses = [c.strip() for c in clauses if isinstance(c, str) and c.strip()]
    except Exception as e:
        flash(f"Clause extraction failed: {e}")
        return redirect(url_for("index"))

    if not clauses:
        flash("No clauses could be extracted.")
        return redirect(url_for("index"))
    clause_extraction_time= datetime.now()
    print("time taken to extract the clauses:", clause_extraction_time - file_save)
    print("clauses extracted:", len(clauses))
    results = predict_clauses(clauses)

    summary = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for r in results:
        summary[r["label"]] = summary.get(r["label"], 0) + 1
    summary_time= datetime.now()
    print("time taken to predict the clauses:", summary_time - clause_extraction_time)

    return render_template(
        "results.html",
        filename=filename,
        results=results,
        summary=summary,
        total=len(results),
        results_json=json.dumps(results),
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
