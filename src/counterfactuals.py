from __future__ import annotations
import re
import string
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CF2_PREFIX = "In my answer, I think that"
CF3_SUFFIX = "This is my final answer."

@dataclass
class CounterfactualGenerator:
    train_answers: Iterable[str]

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.vectorizer.fit([str(x) for x in self.train_answers])
        self.feature_names = np.asarray(self.vectorizer.get_feature_names_out())

    @staticmethod
    def cf1(answer: str) -> str:
        lowered = str(answer).lower()
        return lowered.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def cf2(answer: str) -> str:
        return f"{CF2_PREFIX} {str(answer)}".strip()

    @staticmethod
    def cf3(answer: str) -> str:
        return f"{str(answer).strip()} {CF3_SUFFIX}".strip()

    def cf4(self, answer: str) -> str:
        text = str(answer)
        row = self.vectorizer.transform([text])
        indices = row.indices
        values = row.data
        if len(indices) == 0:
            return text
        # Eligible tokens are exactly vocabulary tokens receiving non-zero TF-IDF.
        pairs = [(self.feature_names[i], float(v)) for i, v in zip(indices, values) if v > 0]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        selected = [token for token, _ in pairs[:2]]
        return (text.rstrip() + " " + " ".join(selected)).strip()

    def all(self, answer: str) -> dict[str, str]:
        return {"original": str(answer), "cf1": self.cf1(answer), "cf2": self.cf2(answer), "cf3": self.cf3(answer), "cf4": self.cf4(answer)}
