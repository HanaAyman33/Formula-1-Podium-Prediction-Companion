# Milestone 2: Formula 1 Knowledge Graph (Neo4j)

## Submitted files
- `Create_kg.py`: builds the graph from the 9 CSV files in this folder
- `config.txt`: Neo4j credentials (`URI`, `USERNAME`, `PASSWORD`); not in git, copy `config.example.txt`
- `queries.txt`: the 5 verification queries, generated from `queries/` (do not edit by hand)

## Layout
| Path | What it is |
| --- | --- |
| `Create_kg.py` | loader: constraints, entity nodes, then Result, Qualifying and PitStop facts |
| `*.csv` | the 9 dataset tables (must stay next to `Create_kg.py`) |
| `queries/q1.cypher` ... `q5.cypher` | one query per file, the only place queries are edited |
| `queries.txt` | built from `queries/` by `tools/build_queries.py` |
| `tools/check.py` | checks the built graph: counts, types, links, constraints, `--rerun` for duplicates |
| `tools/verify_queries.py` | runs the 5 queries plus 11 unseen-style queries and compares them with pandas |
| `tools/expected_answers.txt` | the pandas answers (`python tools/verify_queries.py --expected`) |
| `tools/check_entities.cypher` | quick manual checks to paste into Neo4j Query |

## Run (from this folder)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.txt config.txt            # then set your password
python Create_kg.py                         # builds the graph (about 1 minute)
python tools/check.py --rerun               # must end with "all ... checks passed"
python tools/verify_queries.py              # must end with "16 of 16 queries agree"
python tools/build_queries.py               # after editing anything in queries/
```

Expected graph: 51,113 nodes and 163,510 relationships.