import os, re
import pandas as pd
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SEG          = os.path.join(PROJECT_ROOT, "data", "segmented")

df = pd.read_csv(os.path.join(SEG, "clauses_balanced.csv"))
print(f"Loaded {len(df)} clauses")

# ── Keyword lists (reused from Step 6) ──────────────────────
MODALS = {
    "must":0.9,"shall":0.9,"will":0.8,"required":0.8,
    "should":0.6,"may":0.3,"can":0.3,"encouraged":0.1,"optional":0.1
}
CONSEQUENCES = {
    "termination":1.0,"dismissal":1.0,"criminal":0.95,"imprisonment":1.0,
    "prosecution":1.0,"damages":0.85,"penalty":0.85,"fine":0.85,
    "suspension":0.75,"prohibited":0.7,"disciplinary":0.7,
    "breach":0.8,"liable":0.85,"liability":0.85,"sanction":0.85,
    "warning":0.5,"deduction":0.45,"comply":0.4,"obligation":0.45
}
CONDITIONALS = {
    "failure to":1.0,"violation of":1.0,"breach of":1.0,
    "result in":0.85,"constitute":0.8,"shall not":0.85,
    "must not":0.85,"in the event of":0.85,"without consent":0.85,
    "may result":0.6,"provided that":0.6,"unless":0.5
}
NEGATIONS = ["must not","shall not","not permitted","not allowed",
             "without consent","without authorization","strictly prohibited"]

def extract_features(text):
    t   = text.lower()
    tok = [x.text.lower() for x in nlp(t)]

    modal = max((MODALS[w] for w in tok if w in MODALS), default=0.0)

    cons  = 0.0
    for p,w in sorted(CONSEQUENCES.items(), key=lambda x:-len(x[0])):
        if p in t: cons = max(cons, w); break

    cond  = 0.0
    for p,w in sorted(CONDITIONALS.items(), key=lambda x:-len(x[0])):
        if p in t: cond = max(cond, w); break

    neg   = int(any(n in t for n in NEGATIONS))
    wc    = len(text.split())

    # Obligation strength — counts must/shall/will tokens
    oblig = sum(1 for w in tok if w in {"must","shall","will","required"})

    # Penalty flag — binary: does clause contain any penalty keyword
    pflag = int(cons >= 0.7)

    return modal, cons, cond, neg, wc, oblig, pflag

print("Extracting features...")
rows = [extract_features(str(t)) for t in tqdm(df["raw_text"])]
cols = ["modal_score","consequence_score","conditional_score",
        "has_negation","word_count_f","obligation_count","penalty_flag"]
df[cols] = pd.DataFrame(rows, index=df.index)

out = os.path.join(SEG, "clauses_features.csv")
df.to_csv(out, index=False)
print(f"✓ Saved → {out}")
print(df[cols].describe().round(3).to_string())