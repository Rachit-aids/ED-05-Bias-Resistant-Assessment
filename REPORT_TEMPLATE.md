# ED-05 Experimental Report

## 1. Model

- Base checkpoint:
- Maximum sequence length:
- Epochs:
- Learning rate:
- Batch size:
- Seed:

## 2. Development split

Question-grouped split was used so responses to the same question are not split across train and development partitions.

## 3. Results

| Metric | Value |
|---|---:|
| Macro-F1 | |
| CF1 consistency | |
| CF2 consistency | |
| CF3 consistency | |
| Mean CF1–CF3 consistency | |
| CF4 resistance utility | |
| Normalized Brier | |
| Calibration utility | |

## 4. Evidence examples

Use `outputs/evaluation/evidence.json`. Explain that leave-one-out effects are model-sensitivity evidence, not causal explanations.

## 5. Bias / limitations

Discuss domain shift, reference-answer dependence, class imbalance, annotation ambiguity, style coverage, keyword-specific robustness, and the limitations of token-level evidence.

## 6. Reproducibility

Report Python version, package versions, hardware, seed, model checkpoint, and the exact command used for training/evaluation.
