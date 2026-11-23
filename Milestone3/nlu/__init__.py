"""Member 1: intent classification and entity extraction.

    from nlu import classify, extract, missing_slots, missing_question, to_params

    intent = classify(msg)                 # IntentResult: .label is one of the 10 intents,
                                           # "follow_up" (keep the cached intent) or "unclear" (ask to rephrase)
    ex = extract(msg, intent.label)        # Extraction: .entities, .ambiguous, .references, .question()
    # Member 4 merges ex.entities into the session cache, then:
    missing_question(intent, merged)       # None, or the clarifying question to ask
    to_params(merged)                      # flat Cypher parameters for Member 2's templates
"""
import os

from .entities import complete, extract, missing_question, missing_slots, to_params   # noqa: F401
from .intents import INTENTS, LABELS                                                  # noqa: F401
from .rules import IntentResult, classify_rules                                       # noqa: F401

# Which classifier the app uses: "hybrid" (default), "rules" or "embedding".
# hybrid = the rules decide when they match; when they find nothing ("unclear"), the embedding
# classifier gets a chance; if that is unsure too, the user is asked to rephrase.
DEPLOYED = os.environ.get("INTENT_CLASSIFIER", "hybrid")
_embedding = None


def _emb():
    global _embedding
    if _embedding is None:
        from .embedding import EmbeddingClassifier
        _embedding = EmbeddingClassifier()
    return _embedding


def classify_hybrid(text: str) -> IntentResult:
    r = classify_rules(text)
    if r.label != "unclear":
        return IntentResult(r.label, "hybrid", r.scores, f"rules: {r.reason}")
    e = _emb().classify(text)
    return IntentResult(e.label, "hybrid", e.scores, f"rules found nothing, embedding: {e.reason}")


def classify(text: str) -> IntentResult:
    if DEPLOYED == "rules":
        return classify_rules(text)
    if DEPLOYED == "embedding":
        return _emb().classify(text)
    return classify_hybrid(text)
