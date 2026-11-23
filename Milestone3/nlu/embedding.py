"""Neural intent classifier: embedding similarity (Milestone 3, section 1.a).

The example phrasings of every intent (intents.py) are embedded once (and cached on disk). A message
is embedded and compared by cosine similarity with every example; an intent's score is its
best-matching example. Before embedding, names and years are replaced by their type
("How many points did Hamilton score in 2019?" -> "how many points did driver score in year"), so
the similarity reflects the question being asked and not which driver, race or season it names.
Decision, in order:

    follow-up ("And in 2020?")       -> "follow_up": the session keeps the cached intent
    nothing about F1 in the message  -> out_of_scope   (rule signal: no F1 term, name, team, circuit, race)
    asks for data the graph lacks    -> out_of_scope   (rule signal: salary, tyres, sprints, lap times ...)
    best score < MIN_SCORE           -> "unclear"      (ask the user to rephrase)
    best - second < MARGIN           -> "unclear"      (two intents too close to call)
    nearest intent is out_of_scope   -> "unclear"      (similarity alone never marks a message out of scope)
    otherwise                        -> the nearest intent
"""
import hashlib
import json
import os

import re

import numpy as np

from .intents import INTENTS
from .rules import IntentResult, domain_evidence, is_follow_up, unanswerable
from .entities import extract

MODEL_NAME = os.environ.get("INTENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MIN_SCORE = float(os.environ.get("INTENT_MIN_SCORE", 0.50))  # chosen with eval/calibrate_embedding.py
MARGIN = float(os.environ.get("INTENT_MARGIN", 0.02))
PREDICT_CUE = re.compile(r"\bpredict|\bpodium\b|\btop (three|3)\b")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


class SentenceEncoder:
    """Wraps sentence-transformers; returns L2-normalised vectors so a dot product is the cosine."""

    def __init__(self, name=MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name)

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


class EmbeddingClassifier:
    def __init__(self, encoder=None, min_score=MIN_SCORE, margin=MARGIN):
        self.encoder = encoder or SentenceEncoder()
        self.min_score, self.margin = min_score, margin
        self.labels, texts = [], []
        for label, spec in INTENTS.items():
            for t in spec["examples"]:
                self.labels.append(label)
                texts.append(extract(t).masked())
        self.labels = np.array(self.labels)
        self.example_vecs = self._cached_encode(texts)        # embedded once, then read from disk

    def _cached_encode(self, texts):
        name = getattr(self.encoder, "name", type(self.encoder).__name__)
        key = hashlib.sha1(json.dumps([name, texts]).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, f"intent_examples_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        vecs = self.encoder.encode(texts)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def scores(self, text, ex=None):
        ex = ex or extract(text)
        sims = self.example_vecs @ self.encoder.encode([ex.masked()])[0]
        return {label: float(sims[self.labels == label].max()) for label in INTENTS}

    def classify(self, text) -> IntentResult:
        ex = extract(text)
        if is_follow_up(text, ex):
            return IntentResult("follow_up", "embedding", reason="no new intent named: keep the cached intent")
        sc = self.scores(text, ex)
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        (top, s1), (_, s2) = ranked[0], ranked[1]
        about_f1 = domain_evidence(text, ex)
        cannot = unanswerable(text)

        if not about_f1:
            # the second signal: nothing in the message is about F1 (no term, driver, team, circuit, race)
            return IntentResult("out_of_scope", "embedding", sc, "no Formula 1 term or name in the message")
        if cannot and not PREDICT_CUE.search(text.lower()):
            # a rule signal, not similarity: the message asks for data the graph does not hold
            return IntentResult("out_of_scope", "embedding", sc, "asks for data the graph does not hold")
        if s1 < self.min_score:
            return IntentResult("unclear", "embedding", sc, f"best score {s1:.2f} below {self.min_score}")
        if s1 - s2 < self.margin:
            return IntentResult("unclear", "embedding", sc, f"{top} and {ranked[1][0]} are within {self.margin}")
        if top == "out_of_scope":
            return IntentResult("unclear", "embedding", sc, "nearest to out-of-scope, but no second signal: ask to rephrase")
        return IntentResult(top, "embedding", sc, f"nearest intent {s1:.2f}, margin {s1 - s2:.2f}")
