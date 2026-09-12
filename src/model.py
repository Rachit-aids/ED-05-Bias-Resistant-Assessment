from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["correct", "contradictory", "incorrect"]

class ShortAnswerModel:
    def __init__(self, model_name_or_dir: str, max_length: int = 384, device: str | None = None):
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_dir, num_labels=3, ignore_mismatched_sizes=False)
        self.model.to(self.device)
        self.model.eval()
        self.temperature = 1.0
        temp_file = Path(model_name_or_dir) / "temperature.json"
        if temp_file.exists():
            self.temperature = float(json.loads(temp_file.read_text()) ["temperature"])

    @staticmethod
    def pack(question: str, reference: str, answer: str) -> str:
        return f"Question: {question}\nReference answer: {reference}\nStudent answer: {answer}"

    def logits(self, questions, references, answers, batch_size=16):
        all_logits = []
        for start in range(0, len(answers), batch_size):
            texts = [self.pack(q, r, a) for q, r, a in zip(questions[start:start+batch_size], references[start:start+batch_size], answers[start:start+batch_size])]
            enc = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            all_logits.append(out.logits.detach().cpu())
        return torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, 3))

    def predict_proba(self, questions, references, answers, batch_size=16):
        logits = self.logits(questions, references, answers, batch_size=batch_size) / self.temperature
        return torch.softmax(logits, dim=-1).numpy()
