from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_table, LABEL_TO_ID
from src.model import ShortAnswerModel
from src.counterfactuals import CounterfactualGenerator
from src.metrics import macro_f1, normalized_brier, cf_consistency, cf4_utility, report, confusion
from src.evidence import token_leave_one_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--tfidf-train", required=True, help="Development training CSV used to fit the exact CF4 TF-IDF vectorizer")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--evidence-n", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_table(args.input, require_label=True)
    tfidf_train = load_table(args.tfidf_train, require_label=True)
    model = ShortAnswerModel(args.model_dir)
    cf = CounterfactualGenerator(tfidf_train.student_answer.tolist())
    originals = model.predict_proba(df.question.tolist(), df.reference_answer.tolist(), df.student_answer.tolist(), args.batch_size)
    cf_probs = {}
    for typ in ("cf1","cf2","cf3","cf4"):
        answers = [cf.all(a)[typ] for a in df.student_answer]
        cf_probs[typ] = model.predict_proba(df.question.tolist(), df.reference_answer.tolist(), answers, args.batch_size)
    y = df.label.map(LABEL_TO_ID).to_numpy()
    metrics = {
        "macro_f1": macro_f1(y, originals),
        "brier": normalized_brier(y, originals),
        "calibration_utility": 1.0 - normalized_brier(y, originals),
        "cf1_consistency": cf_consistency(originals, cf_probs["cf1"]),
        "cf2_consistency": cf_consistency(originals, cf_probs["cf2"]),
        "cf3_consistency": cf_consistency(originals, cf_probs["cf3"]),
        "cf1_cf3_consistency": sum(cf_consistency(originals, cf_probs[x]) for x in ("cf1","cf2","cf3")) / 3.0,
        "cf4_utility": cf4_utility(originals, cf_probs["cf4"]),
        "classification_report": report(y, originals),
        "confusion_matrix": confusion(y, originals),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rows=[]
    for i, row in df.iterrows():
        d={"question_id":row.question_id,"student_answer":row.student_answer,"gold":row.label,"original_pred":int(originals[i].argmax())}
        for j, typ in enumerate(("original","cf1","cf2","cf3","cf4")):
            p = originals[i] if typ=="original" else cf_probs[typ][i]
            d[f"{typ}_label"]=["correct","contradictory","incorrect"][int(p.argmax())]
            d[f"{typ}_p_correct"]=float(p[0]); d[f"{typ}_p_contradictory"]=float(p[1]); d[f"{typ}_p_incorrect"]=float(p[2])
        rows.append(d)
    pd.DataFrame(rows).to_csv(out / "counterfactual_predictions.csv", index=False)
    evidence=[]
    for i in range(min(args.evidence_n, len(df))):
        e=token_leave_one_out(model, df.iloc[i].question, df.iloc[i].reference_answer, df.iloc[i].student_answer)
        evidence.append({"question_id":df.iloc[i].question_id,"student_answer":df.iloc[i].student_answer,"predicted_label":["correct","contradictory","incorrect"][int(originals[i].argmax())],"evidence":e})
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
