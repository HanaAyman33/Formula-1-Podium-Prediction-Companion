# Member 1: intent classification and entity extraction

## Files
| Path | What it is |
| --- | --- |
| `nlu/intents.py` | The 10 intents: description and the example phrasings the embedding classifier is built from |
| `nlu/rules.py` | Rule-based classifier (keyword and paraphrase patterns plus entity evidence), and the checks both classifiers share: F1 evidence, unanswerable topics, follow-up detection |
| `nlu/embedding.py` | Embedding-similarity classifier (`all-MiniLM-L6-v2`): names and years are masked before embedding; minimum score 0.50, margin 0.02 |
| `nlu/__init__.py` | Entry point. `classify()` uses the **hybrid** by default: rules first, the embedding classifier when the rules find nothing |
| `nlu/entities.py` | Entity extraction, clarifying questions, required slots per intent, Cypher parameters |
| `nlu/lexicon.py` | Name tables built from the CSVs: drivers, team aliases, circuits by name or city, races |
| `data/intent_eval_set.csv` | Labelled set: 70 questions, 7 per intent, worded differently from the examples |
| `eval/eval_intents.py` | Accuracy, confusion matrix and error list for the rules, embedding and hybrid classifiers |
| `eval/calibrate_embedding.py` | Chooses the minimum score and margin on the examples only (leave-one-out) |
| `tests/test_nlu.py` | One check per required behaviour |

The CSVs (`drivers`, `constructors`, `circuits`, `races`, `results`) are read from `data/`. Set
`F1_DATA_DIR` if they live elsewhere.

## Run
```bash
pip install -r requirements-nlu.txt
python tests/test_nlu.py                 # 14 checks
python eval/calibrate_embedding.py       # pick INTENT_MIN_SCORE / INTENT_MARGIN
python eval/eval_intents.py              # rules, embedding, hybrid -> eval/results/
```
The first embedding run downloads the MiniLM model (about 90 MB) and caches the example vectors in `.cache/`.

## Interface for the other members
```python
from nlu import classify, extract, missing_question, to_params

r = classify(msg)                 # r.label: one of the 10 intents, "follow_up" or "unclear"
ex = extract(msg, r.label)        # ex.entities, ex.ambiguous, ex.references ("driver" for he/his)
ex.question()                     # a clarifying question for an ambiguous name, or None
# Member 4: merge ex.entities into the session cache, then
missing_question(intent, merged)  # None, or the question for a missing slot
to_params(merged)                 # driver_id, driver2_id, constructor_id, circuit_id, race_id,
                                  # race_date, season, year_from, year_to, top_n  -> Member 2
```

## Results (70-question labelled set)
| Classifier | Accuracy | Wrong intent | Asks to rephrase |
| --- | --- | --- | --- |
| Rules | 94.3% | 2 | 2 |
| Embedding, before masking | 62.9% | 5 | 17 |
| Embedding, after masking | 85.7% | 4 | 6 |
| **Hybrid (deployed)** | **95.7%** | **3** | **0** |

Thresholds were chosen with `eval/calibrate_embedding.py` (leave-one-out on the examples), not on the
labelled set. Masking names and years raised the leave-one-out accuracy of the examples from 42.5%
to 73.3%. Some rule patterns were added after a first look at the labelled set, so the rule and
hybrid scores are somewhat optimistic; a fresh set of unseen questions gives the fairest number.

## Decisions
- **Hybrid in the app:** the rules are precise and fast when they match; the embedding classifier covers wordings the rules do not know. The cost: the hybrid never asks to rephrase, so a question the rules cannot classify gets the embedding's best guess.
- **Names and years are masked before embedding** ("who won the year grand prix"), so similarity reflects the question and not the race or season it names.
- **Out of scope is never decided by similarity alone.** The embedding classifier sends a message to `out_of_scope` only on a rule signal: there is nothing about F1 in it (no F1 term, driver, team, circuit or race), or it asks for data the graph does not hold (salary, tyres, sprints, lap times and similar). When the nearest examples are the out-of-scope ones but there is no such signal, it asks the user to rephrase.
- **Follow-ups** such as "And in 2020?", "What about Norris?" or "2021?" return `follow_up`, so the cached intent is kept. "How many points did he score in 2019?" keeps its own intent and reports the reference "driver", so the cached driver is used.
- **Ambiguous names are never guessed.** "Schumacher" is a question. A season in the same message narrows it ("Schumacher in 2021" is Mick), because only one candidate raced that year. "Lotus" (three teams) and "Las Vegas" (two circuits) are asked about the same way.
- **A team name that is also a surname** ("McLaren", "Williams") is read as the team, unless the intent needs more drivers than are named and nothing marks it as a team ("at", "for", "team").
- **Typos** in names of five or more letters are matched (for example "Verstapen"). The match must keep the first letter and roughly the same length, and a typo-matched name alone never counts as evidence that the message is about F1.
- **"Before 2020"** means seasons up to 2019, matching races dated before 2020-01-01.
