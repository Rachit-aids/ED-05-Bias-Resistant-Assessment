from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_table, LABELS
from src.model import ShortAnswerModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    df = load_table(args.input, require_label=False)
    model = ShortAnswerModel(args.model_dir)
    p = model.predict_proba(df.question.tolist(), df.reference_answer.tolist(), df.student_answer.tolist(), args.batch_size)
    out = pd.DataFrame({
        "question_id": df.question_id,
        "predicted_label": [LABELS[i] for i in p.argmax(axis=1)],
        "p_correct": p[:,0],
        "p_contradictory": p[:,1],
        "p_incorrect": p[:,2],
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

if __name__ == "__main__": main()
