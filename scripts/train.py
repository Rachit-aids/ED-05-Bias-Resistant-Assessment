from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.seed import set_seed
from src.data import load_table, LABEL_TO_ID, LABELS
from src.counterfactuals import CounterfactualGenerator
from src.calibration import fit_temperature
from src.metrics import macro_f1, normalized_brier


def pack(q, r, a):
    return f"Question: {q}\nReference answer: {r}\nStudent answer: {a}"

def batch_encode(tokenizer, texts, max_length, device):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)

def kl_symmetric(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    return 0.5 * (F.kl_div(p, q_prob, reduction="batchmean") + F.kl_div(q, p_prob, reduction="batchmean"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="roberta-base")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.03)
    ap.add_argument("--cf-consistency-weight", type=float, default=0.25)
    ap.add_argument("--cf4-hinge-weight", type=float, default=0.20)
    ap.add_argument("--cf4-margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    train = load_table(args.train, require_label=True)
    dev = load_table(args.dev, require_label=True)
    train = train.reset_index(drop=True); dev = dev.reset_index(drop=True)
    cf = CounterfactualGenerator(train["student_answer"].tolist())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.config.id2label = {i: x for i, x in enumerate(LABELS)}
    model.config.label2id = {x: i for i, x in enumerate(LABELS)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    counts = train["label"].map(LABEL_TO_ID).value_counts().reindex([0,1,2]).fillna(1).values
    weights = np.sqrt(counts.sum() / counts)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.gradient_accumulation)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    def make_batch(df):
        rows = df.sample(frac=1.0, random_state=np.random.randint(0, 2**31-1)).reset_index(drop=True)
        for s in range(0, len(rows), args.batch_size):
            b = rows.iloc[s:s+args.batch_size]
            q, r, a = b.question.tolist(), b.reference_answer.tolist(), b.student_answer.tolist()
            variants = [cf.all(x) for x in a]
            yield b, q, r, a, variants

    best = -1.0
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); running = []
        for step, (b, q, r, a, variants) in enumerate(make_batch(train)):
            texts = []
            for i in range(len(a)):
                texts.append(pack(q[i], r[i], variants[i]["original"]))
            for typ in ("cf1", "cf2", "cf3", "cf4"):
                for i in range(len(a)):
                    texts.append(pack(q[i], r[i], variants[i][typ]))
            enc = batch_encode(tokenizer, texts, args.max_length, device)
            logits = model(**enc).logits
            n = len(a)
            orig = logits[:n]
            cf1, cf2, cf3, cf4 = logits[n:2*n], logits[2*n:3*n], logits[3*n:4*n], logits[4*n:5*n]
            labels = torch.tensor([LABEL_TO_ID[x] for x in b.label.tolist()], device=device)
            ce = F.cross_entropy(orig, labels, weight=weights, label_smoothing=args.label_smoothing)
            consistency = (kl_symmetric(orig, cf1) + kl_symmetric(orig, cf2) + kl_symmetric(orig, cf3)) / 3.0
            p0 = F.softmax(orig, dim=-1)[:, 0]
            p4 = F.softmax(cf4, dim=-1)[:, 0]
            hinge = F.relu(p4 - p0 - args.cf4_margin).mean()
            loss = ce + args.cf_consistency_weight * consistency + args.cf4_hinge_weight * hinge
            (loss / args.gradient_accumulation).backward()
            running.append(float(loss.detach().cpu()))
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == math.ceil(len(train) / args.batch_size):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        
        model.eval()
        dev_logits = []
        with torch.no_grad():
            for s in range(0, len(dev), args.batch_size):
                b = dev.iloc[s:s+args.batch_size]
                enc = batch_encode(tokenizer, [pack(q,r,a) for q,r,a in zip(b.question,b.reference_answer,b.student_answer)], args.max_length, device)
                dev_logits.append(model(**enc).logits.detach().cpu())
        dev_logits = torch.cat(dev_logits).numpy()
        y = dev.label.map(LABEL_TO_ID).to_numpy()
        probs = torch.softmax(torch.tensor(dev_logits), dim=-1).numpy()
        f1 = macro_f1(y, probs)
        brier = normalized_brier(y, probs)
        print(f"epoch={epoch+1} loss={np.mean(running):.4f} dev_macro_f1={f1:.4f} dev_brier={brier:.4f}")
        if f1 > best:
            best = f1
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            np.save(out / "dev_logits.npy", dev_logits)
            (out / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    # Temperature calibration is fit on the selected model's development logits.
    best_model = AutoModelForSequenceClassification.from_pretrained(out).to(device).eval()
    logits = np.load(out / "dev_logits.npy")
    y = dev.label.map(LABEL_TO_ID).to_numpy()
    temperature = fit_temperature(logits, y)
    (out / "temperature.json").write_text(json.dumps({"temperature": temperature}, indent=2))
    print(f"saved={out} best_dev_macro_f1={best:.4f} temperature={temperature:.5f}")

if __name__ == "__main__":
    main()
