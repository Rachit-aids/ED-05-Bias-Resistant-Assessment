from __future__ import annotations
import numpy as np

def token_leave_one_out(model, question, reference, answer, top_k=8):
    tokens = answer.split()
    if not tokens:
        return []
    base = model.predict_proba([question], [reference], [answer])[0]
    pred = int(np.argmax(base))
    effects = []
    for i, tok in enumerate(tokens):
        altered = " ".join(tokens[:i] + tokens[i+1:]).strip()
        p = model.predict_proba([question], [reference], [altered])[0]
        effects.append({"token": tok, "index": i, "predicted_class": pred, "delta_predicted_probability": float(p[pred] - base[pred]), "delta_correct_probability": float(p[0] - base[0])})
    effects.sort(key=lambda x: abs(x["delta_predicted_probability"]), reverse=True)
    return effects[:top_k]
