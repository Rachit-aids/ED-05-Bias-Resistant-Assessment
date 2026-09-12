# ED-05 — Bias-Resistant Short-Answer Assessment

A reproducible three-class short-answer assessment system for SemEval-2013 Task 7. The model predicts:

- `correct`
- `contradictory`
- `incorrect`

The implementation is reference-aware: it jointly encodes the question/context, reference answer, and student answer with a transformer cross-encoder. Robustness is trained and evaluated with the exact counterfactual families specified by ED-05.

## Design

1. **Semantic classifier:** transformer sequence classifier over `question + reference answer + student answer`.
2. **Style invariance:** CF1–CF3 are generated locally and used with a symmetric KL consistency penalty during training.
3. **Keyword repetition resistance:** CF4 follows the exact TF-IDF rule in the brief. Training optionally adds a hinge penalty when CF4 raises `P(correct)` above the original.
4. **Calibration:** temperature scaling is fitted only on the local development split.
5. **Evaluation:** macro-F1, CF1–CF3 consistency, CF4 resistance utility, normalized multiclass Brier score, calibration utility, and per-class metrics.
6. **Evidence:** token-level leave-one-out evidence identifies student-answer tokens whose removal changes the predicted class probability most. This is diagnostic evidence, not a causal explanation.

## Dataset

Use the public SemEval-2013 Task 7 three-way data from the official dataset repository:

https://github.com/myrosia/semeval-2013-task7

The repository contains the `semeval-3way.zip` resource and is licensed CC-BY-SA. See the original task paper:

https://aclanthology.org/S13-2045/

Do **not** include evaluation labels in an evaluator submission. This project never looks up evaluation labels during inference.

## Quick start

### 1. Environment

Python 3.10+ is recommended. Install:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and prepare development data

```bash
python scripts/download_data.py
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed
```

`prepare_data.py` accepts the original XML/TSV/CSV files found in the SemEval archive and normalizes them to:

```text
question_id,question,reference_answer,student_answer,label,split,corpus
```

The normalizer deliberately keeps only the three-way labels and does not create any answer-to-label lookup table.

### 3. Train

```bash
python scripts/train.py \
  --train data/processed/train.csv \
  --dev data/processed/dev.csv \
  --output-dir models/ed05 \
  --model-name roberta-base \
  --epochs 3 \
  --batch-size 8 \
  --gradient-accumulation 2
```

For a CPU smoke test, use:

```bash
python scripts/train.py --train data/sample_train.csv --dev data/sample_dev.csv --output-dir models/smoke --model-name distilroberta-base --epochs 1 --batch-size 2
```

### 4. Predict

Input CSV columns:

```text
question_id,question,reference_answer,student_answer
```

Run:

```bash
python scripts/predict.py \
  --model-dir models/ed05 \
  --input evaluator_items.csv \
  --output outputs/predictions.csv
```

Output includes:

```text
question_id,predicted_label,p_correct,p_contradictory,p_incorrect
```

No gold labels are required or read by inference.

### 5. Robustness evaluation

For a labeled local evaluation file:

```bash
python scripts/evaluate.py \
  --model-dir models/ed05 \
  --input data/processed/dev.csv \
  --output-dir outputs/evaluation
```

This generates originals, CF1–CF4 predictions, macro-F1, consistency, CF4 utility, normalized Brier, calibration utility, and an evidence file.

## Exact counterfactual definitions

### CF1
Lowercase the entire answer and remove punctuation.

### CF2
Prepend exactly:

`In my answer, I think that`

### CF3
Append exactly:

`This is my final answer.`

### CF4
Fit `sklearn.feature_extraction.text.TfidfVectorizer` on development training answers using:

```python
TfidfVectorizer(lowercase=True, stop_words="english")
```

with the default token pattern. For each current answer, select up to two eligible tokens with largest TF-IDF values, breaking ties lexicographically, and append them once each. If fewer than two eligible tokens exist, append all eligible tokens once.

The implementation in `src/counterfactuals.py` follows this literally.

## Metrics

- **Macro-F1:** `sklearn.metrics.f1_score(..., average="macro")`
- **CF1–CF3 consistency:** fraction whose predicted class matches the corresponding original prediction.
- **CF4 resistance:**

  `1 - mean(max(0, P_correct(CF4) - P_correct(original)))`, clipped to `[0, 1]`.

- **Normalized multiclass Brier:**

  `1/(2N) * sum_i sum_k (p_ik - y_ik)^2`

- **Calibration utility:** `1 - Brier`

## Reproducibility

- Global random seeds are set for Python, NumPy and PyTorch.
- Data splitting is grouped by `question_id` to reduce leakage between near-duplicate responses to the same question.
- The training configuration is saved beside the model.
- The temperature used for calibration is saved in `temperature.json`.
- Dependencies are pinned in `requirements.txt`.

## Important anti-leakage rules

The evaluation side must not:

- load public SemEval gold labels;
- query a public copy for labels;
- reconstruct labels from answer IDs;
- fingerprint evaluator IDs against the corpus;
- use a memorized answer-to-label dictionary.

The inference script therefore requires only question/context, reference answer, and student answer.

## Limitations and bias analysis

1. **Domain shift:** SciEntsBank and Beetle are educational science datasets; performance can degrade on new subjects, grade levels, or answer styles.
2. **Reference-answer dependence:** a response can be semantically valid but receive a poor score if the reference does not express the same concept clearly.
3. **Annotation ambiguity:** `incorrect` includes heterogeneous cases after collapsing the five-way task into three labels.
4. **Contradiction sparsity:** contradictory examples are usually less frequent than correct/incorrect examples, so class-wise calibration and macro-F1 matter more than accuracy.
5. **Style bias is reduced, not eliminated:** CF1–CF3 only test a limited set of transformations.
6. **Keyword robustness is targeted:** CF4 explicitly measures one adversarial mechanism; it does not establish general adversarial robustness.
7. **Evidence is diagnostic:** leave-one-out token effects show model sensitivity, not human-interpretable causal reasoning.

## Citation

Dzikovska, M. O., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., & Dang, H. T. (2013). *SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge.*

See: https://aclanthology.org/S13-2045/
