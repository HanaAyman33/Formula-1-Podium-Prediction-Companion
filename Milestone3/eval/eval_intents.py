"""Compare the rule-based and the embedding intent classifiers on the labelled set (section 1.a).

    python eval/eval_intents.py                 # rules, embedding and hybrid (needs the embedding model)
    python eval/eval_intents.py --rules-only    # rule-based only

Writes, for each classifier, to eval/results/:
    intent_<method>_predictions.csv    every question, gold label, predicted label, reason
    intent_<method>_confusion.csv/.png confusion matrix (rows = gold, columns = predicted)
and eval/results/intent_summary.md with accuracy, per-intent accuracy and the "ask to rephrase" rate.

A prediction of "unclear" (the classifier asks the user to rephrase) or "follow_up" counts as wrong
in the accuracy: it is a safe answer, but not the right intent. Its rate is reported separately.
"""
import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nlu.intents import LABELS              # noqa: E402
from nlu.rules import classify_rules        # noqa: E402

OUT = os.path.join(ROOT, "eval", "results")
EXTRA = ["unclear", "follow_up"]


def load_set(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r["intent"] not in LABELS]
    if bad:
        sys.exit(f"unknown labels in the eval set: {sorted({r['intent'] for r in bad})}")
    return rows


def check_set(rows):
    from collections import Counter
    n = Counter(r["intent"] for r in rows)
    problems = [f"{k}: {n.get(k, 0)} questions (need >= 4)" for k in LABELS if n.get(k, 0) < 4]
    if len(rows) < 50:
        problems.append(f"only {len(rows)} questions (need >= 50)")
    from nlu.intents import INTENTS
    examples = {e.strip().lower() for v in INTENTS.values() for e in v["examples"]}
    copied = [r["question"] for r in rows if r["question"].strip().lower() in examples]
    if copied:
        problems.append(f"{len(copied)} eval questions are copies of classifier examples: {copied[:3]}")
    return n, problems


def confusion(gold, pred):
    cols = LABELS + EXTRA
    m = np.zeros((len(LABELS), len(cols)), dtype=int)
    for g, p in zip(gold, pred):
        m[LABELS.index(g), cols.index(p)] += 1
    return m, cols


def plot(m, cols, title, path):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(name, fn, rows):
    t0 = time.perf_counter()
    preds = [fn(r["question"]) for r in rows]
    ms = (time.perf_counter() - t0) * 1000 / len(rows)
    gold = [r["intent"] for r in rows]
    labels = [p.label for p in preds]
    acc = float(np.mean([g == p for g, p in zip(gold, labels)]))
    m, cols = confusion(gold, labels)

    with open(os.path.join(OUT, f"intent_{name}_predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "gold", "predicted", "correct", "reason"])
        for r, p in zip(rows, preds):
            w.writerow([r["question"], r["intent"], p.label, r["intent"] == p.label, p.reason])
    with open(os.path.join(OUT, f"intent_{name}_confusion.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gold \\ predicted"] + cols)
        for lab, row in zip(LABELS, m):
            w.writerow([lab] + list(row))
    plot(m, cols, f"{name}: accuracy {acc:.1%} on {len(rows)} questions",
         os.path.join(OUT, f"intent_{name}_confusion.png"))

    per = {lab: float(np.mean([p == lab for g, p in zip(gold, labels) if g == lab])) for lab in LABELS}
    rephrase = sum(p in EXTRA for p in labels) / len(labels)
    wrong = [(r["question"], r["intent"], p.label) for r, p in zip(rows, preds) if r["intent"] != p.label]
    print(f"\n{name}: accuracy {acc:.1%} ({sum(g == p for g, p in zip(gold, labels))}/{len(rows)}), "
          f"asks to rephrase {rephrase:.1%}, {ms:.1f} ms per question")
    for q, g, p in wrong:
        print(f"   gold {g:<18} predicted {p:<18} {q}")
    return dict(name=name, acc=acc, per=per, rephrase=rephrase, ms=ms, wrong=wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(ROOT, "data", "intent_eval_set.csv"))
    ap.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = load_set(args.set)
    counts, problems = check_set(rows)
    print(f"eval set: {len(rows)} questions; per intent: {dict(counts)}")
    for p in problems:
        print("  WARNING:", p)

    results = [run("rules", classify_rules, rows)]
    if not args.rules_only:
        from nlu.embedding import EmbeddingClassifier, MARGIN, MIN_SCORE, MODEL_NAME
        clf = EmbeddingClassifier()
        results.append(run("embedding", clf.classify, rows))
        import nlu
        nlu._embedding = clf                       # reuse the loaded model
        results.append(run("hybrid", nlu.classify_hybrid, rows))

    lines = ["# Intent classification: rule-based vs embedding similarity", "",
             f"Labelled set: {len(rows)} questions, {min(counts.values())}-{max(counts.values())} per intent, "
             "worded differently from the classifier examples.", ""]
    if not args.rules_only:
        lines += [f"Embedding model: `{MODEL_NAME}`, minimum score {MIN_SCORE}, margin {MARGIN}.", ""]
    lines += ["| Method | Accuracy | Asks to rephrase | ms / question |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['name']} | {r['acc']:.1%} | {r['rephrase']:.1%} | {r['ms']:.1f} |" for r in results]
    lines += ["", "| Intent | " + " | ".join(r["name"] for r in results) + " |",
              "| --- | " + " | ".join("---" for _ in results) + " |"]
    lines += [f"| {lab} | " + " | ".join(f"{r['per'][lab]:.0%}" for r in results) + " |" for lab in LABELS]
    for r in results:
        lines += ["", f"## Errors: {r['name']}", ""]
        lines += [f"- gold `{g}`, predicted `{p}`: {q}" for q, g, p in r["wrong"]] or ["- none"]
    with open(os.path.join(OUT, "intent_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
