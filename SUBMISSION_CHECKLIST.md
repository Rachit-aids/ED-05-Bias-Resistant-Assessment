# ED-05 Submission Checklist

- [x] Three-class classifier: correct / contradictory / incorrect
- [x] Three probability outputs
- [x] Reference-aware semantic input
- [x] CF1–CF3 exact local variants
- [x] CF4 exact sklearn TF-IDF construction
- [x] CF1–CF3 consistency training regularizer
- [x] CF4 correct-probability increase penalty
- [x] Macro-F1
- [x] CF1–CF3 consistency
- [x] CF4 resistance utility
- [x] Normalized multiclass Brier
- [x] Calibration utility / temperature scaling
- [x] Example token-level evidence
- [x] Limitations and bias analysis
- [x] Seed control and grouped question split
- [x] Dependency files
- [x] No evaluation-label lookup logic

## Before uploading to GitHub

1. Do not commit `data/raw/`, `data/processed/`, `models/`, or `outputs/`.
2. Run `pytest -q`.
3. Run the real SemEval training pipeline once.
4. Record the final development metrics in your report.
5. Keep the evaluator input label-free.
6. Do not add an answer-ID mapping or cached gold labels.
