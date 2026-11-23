"""Choose MIN_SCORE and MARGIN for the embedding classifier WITHOUT touching the labelled eval set.

Leave-one-out over the classifier's own examples: each example is classified against all the
other examples. For a grid of (min score, margin) it prints how often the answer is right, wrong,
or "ask to rephrase", so the thresholds are picked on the building data only.

    python eval/calibrate_embedding.py
Then set INTENT_MIN_SCORE / INTENT_MARGIN (environment) or the defaults in nlu/embedding.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlu.embedding import EmbeddingClassifier   # noqa: E402
from nlu.intents import LABELS                   # noqa: E402

clf = EmbeddingClassifier()
V, y = clf.example_vecs, clf.labels
S = V @ V.T
np.fill_diagonal(S, -1.0)                        # an example may not match itself

best_scores = np.stack([S[:, y == lab].max(axis=1) for lab in LABELS], axis=1)
order = np.argsort(-best_scores, axis=1)
top = np.array(LABELS)[order[:, 0]]
s1 = best_scores[np.arange(len(y)), order[:, 0]]
s2 = best_scores[np.arange(len(y)), order[:, 1]]
in_domain = y != "out_of_scope"                  # out-of-scope is decided by the rule signals, not here

print(f"{len(y)} examples, nearest-example accuracy with no thresholds: {np.mean(top == y):.1%}\n")
print(" min   margin   right   wrong   rephrase")
for mn in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
    for mg in (0.00, 0.02, 0.03, 0.05, 0.08):
        ask = (s1 < mn) | (s1 - s2 < mg)
        right = np.mean((~ask & (top == y))[in_domain])
        wrong = np.mean((~ask & (top != y))[in_domain])
        print(f" {mn:.2f}  {mg:.2f}    {right:6.1%}  {wrong:6.1%}  {np.mean(ask[in_domain]):6.1%}")
print("\nPick the row with few wrong answers and a rephrase rate you accept (e.g. under ~10%).")
