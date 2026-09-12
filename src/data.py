from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd

LABELS = ["correct", "contradictory", "incorrect"]
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}
ALIASES = {
    "correct": "correct", "CORRECT": "correct",
    "contradictory": "contradictory", "CONTRADICTORY": "contradictory",
    "incorrect": "incorrect", "INCORRECT": "incorrect",
}

REQUIRED = ["question_id", "question", "reference_answer", "student_answer"]

def normalize_label(value):
    s = str(value).strip()
    if s in ALIASES:
        return ALIASES[s]
    raise ValueError(f"Unsupported three-way label: {value!r}")

def load_table(path: str | Path, require_label: bool = False) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError(f"Unsupported table format: {path}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(df.columns)}")
    for c in REQUIRED:
        df[c] = df[c].fillna("").astype(str)
    if require_label:
        if "label" not in df.columns:
            raise ValueError("A labeled file is required for training/evaluation.")
        df["label"] = df["label"].map(normalize_label)
    return df

def save_normalized(rows, path):
    df = pd.DataFrame(rows)
    cols = ["question_id", "question", "reference_answer", "student_answer", "label", "split", "corpus"]
    for c in cols:
        if c not in df:
            df[c] = ""
    df[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
