"""Member 4: verify the graph built by Create_kg.py.

The expected graph is rebuilt here from the 9 CSV files with its own parsing code (it does not
reuse the loader's helpers), so a bug in a loader cannot hide itself. Checks:

  1. totals: 51,113 nodes and 163,510 relationships
  2. node count per label, and no unexpected or extra labels
  3. relationship count per (start label, type, end label), so a wrong direction or an extra
     relationship type shows up
  4. every relationship points at the right parent (compared with the CSV foreign keys),
     with no duplicates
  5. every node property: right name, right type (integer / float / string / date), right value,
     missing values left out (never stored as \\N or ''), no extra helper properties
  6. Race.year agrees with the Season it is linked to; Cypher-side type checks
  7. the 9 uniqueness constraints exist
  8. with --rerun: runs Create_kg.py a second time and confirms that nothing changed

Usage:
  python tools/check.py            # checks 1-7
  python tools/check.py --rerun    # checks 1-7, then a second build, then the counts again
  python tools/check.py --offline  # no database: checks what the loaders WOULD send, against the CSVs

Exit code 0 means every check passed.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import date

import neo4j.time
from neo4j import GraphDatabase

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file is in tools/)

EXPECTED_NODES = 51113
EXPECTED_RELS = 163510

INT, FLOAT, STR, DATE = "INTEGER", "FLOAT", "STRING", "DATE"


# --------------------------------------------------------------------------------------------
# Expected schema: label -> csv file, key property, {property: (csv column or function, type)}.
# This is the single place to edit if the schema in the PDF names something differently.
# --------------------------------------------------------------------------------------------
def pit_event_id(row):
    return f'{int(row["raceId"])}:{int(row["driverId"])}:{int(row["stop"])}'


NODE_SCHEMA = {
    "Season": ("seasons.csv", "year", {"year": ("year", INT)}),
    "Status": ("status.csv", "statusId", {"statusId": ("statusId", INT), "status": ("status", STR)}),
    "Circuit": ("circuits.csv", "circuitId", {
        "circuitId": ("circuitId", INT), "circuitRef": ("circuitRef", STR),
        "circuit_name": ("name", STR), "circuit_loc": ("location", STR),
        "circuit_country": ("country", STR), "lat": ("lat", FLOAT), "lng": ("lng", FLOAT),
        "alt": ("alt", INT)}),
    "Constructor": ("constructors.csv", "constructorId", {
        "constructorId": ("constructorId", INT), "constructorRef": ("constructorRef", STR),
        "constructor_name": ("name", STR), "constructor_nationality": ("nationality", STR)}),
    "Driver": ("drivers.csv", "driverId", {
        "driverId": ("driverId", INT), "driverRef": ("driverRef", STR), "number": ("number", INT),
        "code": ("code", STR), "forename": ("forename", STR), "surname": ("surname", STR),
        "dob": ("dob", DATE), "nationality": ("nationality", STR)}),
    "Race": ("races.csv", "raceId", {
        "raceId": ("raceId", INT), "year": ("year", INT), "round": ("round", INT),
        "name": ("name", STR), "date": ("date", DATE), "time": ("time", STR)}),
    "Result": ("results.csv", "resultId", {
        "resultId": ("resultId", INT), "number": ("number", INT), "grid": ("grid", INT),
        "position": ("position", INT), "positionText": ("positionText", STR),
        "positionOrder": ("positionOrder", INT), "points": ("points", FLOAT),
        "laps": ("laps", INT), "time": ("time", STR), "milliseconds": ("milliseconds", INT),
        "fastestLap": ("fastestLap", INT), "rank": ("rank", INT),
        "fastestLapTime": ("fastestLapTime", STR), "fastestLapSpeed": ("fastestLapSpeed", FLOAT)}),
    "Qualifying": ("qualifying.csv", "qualifyId", {
        "qualifyId": ("qualifyId", INT), "number": ("number", INT), "position": ("position", INT),
        "q1": ("q1", STR), "q2": ("q2", STR), "q3": ("q3", STR)}),
    "PitStop": ("pit_stops.csv", "eventId", {
        "eventId": (pit_event_id, STR), "stop": ("stop", INT), "lap": ("lap", INT),
        "time": ("time", STR), "duration": ("duration", STR),
        "milliseconds": ("milliseconds", INT)}),
}

# (start label, type, end label, csv file, function giving the start key, csv column of the end key)
REL_SCHEMA = [
    ("Race", "AT_CIRCUIT", "Circuit", "races.csv", lambda r: int(r["raceId"]), "circuitId"),
    ("Race", "IN_SEASON", "Season", "races.csv", lambda r: int(r["raceId"]), "year"),
    ("Result", "FOR_RACE", "Race", "results.csv", lambda r: int(r["resultId"]), "raceId"),
    ("Result", "BY_DRIVER", "Driver", "results.csv", lambda r: int(r["resultId"]), "driverId"),
    ("Result", "FOR_CONSTRUCTOR", "Constructor", "results.csv", lambda r: int(r["resultId"]), "constructorId"),
    ("Result", "HAS_STATUS", "Status", "results.csv", lambda r: int(r["resultId"]), "statusId"),
    ("Qualifying", "FOR_RACE", "Race", "qualifying.csv", lambda r: int(r["qualifyId"]), "raceId"),
    ("Qualifying", "BY_DRIVER", "Driver", "qualifying.csv", lambda r: int(r["qualifyId"]), "driverId"),
    ("Qualifying", "FOR_CONSTRUCTOR", "Constructor", "qualifying.csv", lambda r: int(r["qualifyId"]), "constructorId"),
    ("PitStop", "FOR_RACE", "Race", "pit_stops.csv", pit_event_id, "raceId"),
    ("PitStop", "BY_DRIVER", "Driver", "pit_stops.csv", pit_event_id, "driverId"),
]


# --------------------------------------------------------------------------------------------
# Independent CSV parsing
# --------------------------------------------------------------------------------------------
def read_config(path=None):
    """Read config.txt by key (URI, USERNAME, PASSWORD, optional DATABASE), never by line number."""
    path = path or os.path.join(HERE, "config.txt")
    cfg = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k.strip().upper()] = v.strip()
    missing = [k for k in ("URI", "USERNAME", "PASSWORD") if not cfg.get(k)]
    if missing:
        sys.exit(f"config.txt is missing {missing}")
    return cfg


_csv_cache = {}


def csv_rows(name):
    if name not in _csv_cache:
        with open(os.path.join(HERE, name), encoding="utf-8", newline="") as f:
            _csv_cache[name] = list(csv.DictReader(f))
    return _csv_cache[name]


def parse(raw, typ):
    """CSV text -> the value the graph must hold, or None when the property must be left out."""
    if raw is None or raw.strip() in ("", r"\N"):
        return None
    if typ == INT:
        f = float(raw)
        if not f.is_integer():
            raise ValueError(f"{raw!r} is not an integer")
        return int(f)
    if typ == FLOAT:
        return float(raw)
    if typ == DATE:
        return date.fromisoformat(raw)
    return raw


def expected_nodes(label):
    file, key, props = NODE_SCHEMA[label]
    out = {}
    for row in csv_rows(file):
        node = {}
        for prop, (col, typ) in props.items():
            raw = col(row) if callable(col) else row[col]
            val = parse(raw, typ)
            if val is not None:
                node[prop] = val
        out[node[key]] = node
    return out


def graph_type(v):
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return INT
    if isinstance(v, float):
        return FLOAT
    if isinstance(v, str):
        return STR
    if isinstance(v, neo4j.time.Date) or (isinstance(v, date) and not hasattr(v, "hour")):
        return DATE
    return type(v).__name__


def native(v):
    return v.to_native() if isinstance(v, neo4j.time.Date) else v


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.results = []

    def check(self, name, ok, details=()):
        self.results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for d in list(details)[:15]:
            print(f"         {d}")
        if len(details) > 15:
            print(f"         ... and {len(details) - 15} more")
        return ok

    @property
    def failed(self):
        return [n for n, ok in self.results if not ok]


def q(session, cypher, **params):
    return [r.data() for r in session.run(cypher, **params)]


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------
def snapshot(session):
    labels = {tuple(sorted(r["l"])): r["n"] for r in q(session, "MATCH (n) RETURN labels(n) AS l, count(*) AS n")}
    rels = {(r["a"], r["t"], r["b"]): r["n"] for r in q(
        session, "MATCH (a)-[x]->(b) RETURN labels(a)[0] AS a, type(x) AS t, labels(b)[0] AS b, count(*) AS n")}
    total_nodes = q(session, "MATCH (n) RETURN count(n) AS n")[0]["n"]
    total_rels = q(session, "MATCH ()-[x]->() RETURN count(x) AS n")[0]["n"]
    return total_nodes, total_rels, labels, rels


def check_totals_and_breakdown(session, rep):
    total_nodes, total_rels, labels, rels = snapshot(session)
    rep.check(f"total nodes = {EXPECTED_NODES:,} (found {total_nodes:,})", total_nodes == EXPECTED_NODES)
    rep.check(f"total relationships = {EXPECTED_RELS:,} (found {total_rels:,})", total_rels == EXPECTED_RELS)

    # nodes per label
    print("\n  label          expected     found")
    bad = []
    for label in NODE_SCHEMA:
        exp = len(csv_rows(NODE_SCHEMA[label][0]))
        got = labels.get((label,), 0)
        print(f"  {label:<13}{exp:>9,}{got:>10,}{'' if exp == got else '   <-- differs'}")
        if exp != got:
            bad.append(f"{label}: expected {exp:,}, found {got:,}")
    for lab, n in labels.items():
        if len(lab) != 1 or lab[0] not in NODE_SCHEMA:
            bad.append(f"unexpected label set {list(lab)} on {n:,} nodes")
    print()
    rep.check("node count per label matches the CSV row counts", not bad, bad)

    # relationships per pattern
    expected = Counter()
    for a, t, b, file, _, _ in REL_SCHEMA:
        expected[(a, t, b)] += len(csv_rows(file))
    print("  pattern                                       expected     found")
    bad = []
    for key in sorted(set(expected) | set(rels)):
        e, g = expected.get(key, 0), rels.get(key, 0)
        print(f"  ({key[0]})-[:{key[1]}]->({key[2]})".ljust(46) + f"{e:>9,}{g:>10,}"
              + ("" if e == g else ("   <-- not in schema (wrong direction or name?)" if not e else "   <-- differs")))
        if e != g:
            bad.append(f"({key[0]})-[:{key[1]}]->({key[2]}): expected {e:,}, found {g:,}")
    print()
    rep.check("relationship count per (start)-[:TYPE]->(end) matches, no wrong directions", not bad, bad)


def check_relationship_targets(session, rep):
    bad = []
    for a, t, b, file, start_key, end_col in REL_SCHEMA:
        a_key, b_key = NODE_SCHEMA[a][1], NODE_SCHEMA[b][1]
        exp = Counter((start_key(r), int(r[end_col])) for r in csv_rows(file))
        got = Counter((r["a"], r["b"]) for r in q(
            session, f"MATCH (a:{a})-[:{t}]->(b:{b}) RETURN a.{a_key} AS a, b.{b_key} AS b"))
        missing, extra = exp - got, got - exp
        dup = sum(n - 1 for n in got.values() if n > 1)
        if missing or extra or dup:
            bad.append(f"({a})-[:{t}]->({b}): {sum(missing.values()):,} missing, "
                       f"{sum(extra.values()):,} wrong/extra, {dup:,} duplicated; "
                       f"e.g. missing {list(missing)[:2]} extra {list(extra)[:2]}")
    rep.check("every relationship links the right nodes (vs CSV foreign keys), no duplicates", not bad, bad)


def check_properties(session, rep):
    issues = defaultdict(list)   # (label, prop, kind) -> examples
    for label, (file, key, props) in NODE_SCHEMA.items():
        exp = expected_nodes(label)
        got = {}
        for r in q(session, f"MATCH (n:{label}) RETURN properties(n) AS p"):
            got[r["p"].get(key)] = r["p"]
        for k in set(exp) - set(got):
            issues[(label, key, "node missing")].append(k)
        for k in set(got) - set(exp):
            issues[(label, key, "unexpected node")].append(k)
        for k in set(exp) & set(got):
            e, g = exp[k], got[k]
            for prop in set(g) - set(props):
                issues[(label, prop, "property not in schema (helper/renamed?)")].append(k)
            for prop, (_, typ) in props.items():
                if prop not in e and prop in g:
                    issues[(label, prop, f"should be left out (CSV is missing) but holds {g[prop]!r}")].append(k)
                elif prop in e and prop not in g:
                    issues[(label, prop, "missing from node")].append(k)
                elif prop in e:
                    v = g[prop]
                    if isinstance(v, str) and v.strip() in ("", r"\N"):
                        issues[(label, prop, f"holds placeholder {v!r}")].append(k)
                    elif graph_type(v) != typ:
                        issues[(label, prop, f"type {graph_type(v)}, expected {typ}")].append(k)
                    elif native(v) != e[prop]:
                        issues[(label, prop, "value differs from CSV")].append((k, v, e[prop]))
    details = [f"{lab}.{prop}: {kind} on {len(ex):,} nodes, e.g. {ex[:3]}"
               for (lab, prop, kind), ex in sorted(issues.items(), key=lambda i: (i[0][0], i[0][1]))]
    rep.check("every property has the schema name, the right type and the CSV value; "
              "missing values left out", not details, details)

    # belt and braces: no node anywhere stores a placeholder string
    n_bad = q(session, "MATCH (n) UNWIND keys(n) AS k WITH n[k] AS v "
                       "WHERE v = '\\\\N' OR v = '' RETURN count(*) AS n")[0]["n"]
    rep.check(f"no property anywhere holds \\N or an empty string (found {n_bad})", n_bad == 0)


def check_cypher_types(session, rep):
    """The same type rules, asked of Neo4j itself (the way the hidden queries will see the data)."""
    checks = {
        "Race.date is DATE": "MATCH (r:Race) WHERE NOT r.date IS :: DATE RETURN count(r) AS n",
        "Race.year and Race.round are INTEGER":
            "MATCH (r:Race) WHERE NOT r.year IS :: INTEGER OR NOT r.round IS :: INTEGER RETURN count(r) AS n",
        "Season.year is INTEGER": "MATCH (s:Season) WHERE NOT s.year IS :: INTEGER RETURN count(s) AS n",
        "Driver.dob is DATE (where present)":
            "MATCH (d:Driver) WHERE d.dob IS NOT NULL AND NOT d.dob IS :: DATE RETURN count(d) AS n",
        "Result.points is FLOAT": "MATCH (x:Result) WHERE NOT x.points IS :: FLOAT RETURN count(x) AS n",
        "time columns are STRING (Result.time/fastestLapTime, Qualifying.q1-q3, PitStop.duration/time)":
            "MATCH (n) WHERE (n:Result AND ((n.time IS NOT NULL AND NOT n.time IS :: STRING) OR "
            "(n.fastestLapTime IS NOT NULL AND NOT n.fastestLapTime IS :: STRING))) OR "
            "(n:Qualifying AND ANY(k IN ['q1','q2','q3'] WHERE n[k] IS NOT NULL AND NOT n[k] IS :: STRING)) OR "
            "(n:PitStop AND (NOT n.duration IS :: STRING OR NOT n.time IS :: STRING)) RETURN count(n) AS n",
        "PitStop.eventId is a STRING": "MATCH (p:PitStop) WHERE NOT p.eventId IS :: STRING RETURN count(p) AS n",
        "Race.year equals the year of its Season":
            "MATCH (r:Race)-[:IN_SEASON]->(s:Season) WHERE r.year <> s.year RETURN count(r) AS n",
        "every Race has exactly one AT_CIRCUIT and one IN_SEASON":
            "MATCH (r:Race) WHERE COUNT { (r)-[:AT_CIRCUIT]->() } <> 1 OR COUNT { (r)-[:IN_SEASON]->() } <> 1 "
            "RETURN count(r) AS n",
        "every Result has exactly one FOR_RACE, BY_DRIVER, FOR_CONSTRUCTOR, HAS_STATUS":
            "MATCH (x:Result) WHERE COUNT { (x)-[:FOR_RACE]->() } <> 1 OR COUNT { (x)-[:BY_DRIVER]->() } <> 1 "
            "OR COUNT { (x)-[:FOR_CONSTRUCTOR]->() } <> 1 OR COUNT { (x)-[:HAS_STATUS]->() } <> 1 "
            "RETURN count(x) AS n",
    }
    for name, cypher in checks.items():
        try:
            n = q(session, cypher)[0]["n"]
            rep.check(f"[Cypher] {name} (violations: {n})", n == 0)
        except Exception as e:  # e.g. Neo4j older than 5.9 does not know IS :: TYPE
            rep.check(f"[Cypher] {name}: query failed ({type(e).__name__}: {str(e)[:90]})", False)


def check_constraints(session, rep):
    rows = q(session, "SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties "
                      "RETURN type, labelsOrTypes, properties")
    have = {(r["labelsOrTypes"][0], r["properties"][0]) for r in rows
            if "UNIQUE" in r["type"] and r["labelsOrTypes"] and len(r["properties"]) == 1}
    missing = [f"{lab}.{key}" for lab, (_, key, _) in NODE_SCHEMA.items() if (lab, key) not in have]
    rep.check("a uniqueness constraint exists on every node key", not missing,
              [f"no constraint on {m}" for m in missing])


def check_rerun(session, rep):
    before = snapshot(session)
    print("\nRunning Create_kg.py a second time ...")
    sys.path.insert(0, HERE)
    import Create_kg  # noqa: E402
    Create_kg.main()
    after = snapshot(session)
    diffs = []
    if before[0] != after[0]:
        diffs.append(f"nodes {before[0]:,} -> {after[0]:,}")
    if before[1] != after[1]:
        diffs.append(f"relationships {before[1]:,} -> {after[1]:,}")
    for i, what in ((2, "label"), (3, "pattern")):
        for k in set(before[i]) | set(after[i]):
            if before[i].get(k, 0) != after[i].get(k, 0):
                diffs.append(f"{what} {k}: {before[i].get(k, 0):,} -> {after[i].get(k, 0):,}")
    rep.check("second run of Create_kg.py changes nothing (no duplicates)", not diffs, diffs)


def offline(rep):
    """No database needed: run the loaders against a fake session, capture every batch they would
    send, and check the rows and the Cypher patterns against the expected graph."""
    import re
    sys.path.insert(0, HERE)
    import Create_kg

    sent = []   # (cypher, rows)

    class FakeResult:
        def consume(self):
            return None

    class FakeSession:
        def run(self, cypher, rows=None, **_):
            sent.append((" ".join(cypher.split()), rows or []))
            return FakeResult()

    session = FakeSession()
    for fn in ("load_entities", "load_results", "load_qualifying", "load_pitstops"):
        try:
            getattr(Create_kg, fn)(session)
        except Exception as e:
            rep.check(f"[offline] {fn}() runs", False, [f"{type(e).__name__}: {e}"])
            return

    # nodes: the rows sent to each "MERGE (x:Label {key: row.key}) SET x += row"
    node_rows, rel_rows = defaultdict(dict), Counter()
    problems = []   # (message, example); identical messages are grouped and counted below
    for cypher, rows in sent:
        if "MATCH" not in cypher:
            m = re.search(r"MERGE \(\w+:(\w+) \{(\w+): row\.(\w+)\}\)", cypher)
            if not m:
                problems.append(("unrecognised node query (CREATE instead of MERGE?)", cypher[:90]))
                continue
            label, key = m.group(1), m.group(2)
            for row in rows:
                node_rows[label][row[key]] = {**node_rows[label].get(row[key], {}), **row}
            if "SET" not in cypher and len(NODE_SCHEMA.get(label, ("", "", {"x": 0}))[2]) > 1:
                problems.append((f"{label}: MERGE without SET, other properties are never written", ""))
            continue
        # relationships: map variable -> (label, key, row field) from the MATCH clauses
        var = {v: (lab, key, fld) for v, lab, key, fld in
               re.findall(r"\((\w+):(\w+) \{(\w+): row\.(\w+)\}\)", cypher)}
        for a, t, b in re.findall(r"MERGE \((\w+)\)-\[:(\w+)\]->\((\w+)\)", cypher):
            (la, _, fa), (lb, _, fb) = var[a], var[b]
            for row in rows:
                # a field the rows do not have is null in Neo4j: MATCH finds nothing, no relationship is made
                if row.get(fa) is not None and row.get(fb) is not None:
                    rel_rows[(la, t, lb, row[fa], row[fb])] += 1
        if re.search(r"<-\[", cypher) or "CREATE " in cypher:
            problems.append(("uses CREATE or a left-pointing MERGE", cypher[:90]))

    # compare nodes, property by property
    for label, (file, key, props) in NODE_SCHEMA.items():
        exp, got = expected_nodes(label), node_rows.get(label, {})
        if set(exp) != set(got):
            problems.append((f"{label}: {len(set(exp) - set(got))} nodes not sent, {len(set(got) - set(exp))} unexpected", ""))
        for k in set(exp) & set(got):
            e, g = exp[k], got[k]
            for prop in set(g) | set(e):
                if prop not in props:
                    problems.append((f"{label}.{prop}: not in the schema (renamed or helper property?)", f"{key}={k}"))
                elif prop not in g:
                    problems.append((f"{label}.{prop}: missing", f"{key}={k}"))
                elif prop not in e:
                    problems.append((f"{label}.{prop}: should be left out (missing in CSV)", f"{key}={k} sent {g[prop]!r}"))
                elif graph_type(g[prop]) != props[prop][1] or g[prop] != e[prop]:
                    problems.append((f"{label}.{prop}: sent as {graph_type(g[prop])}, expected {props[prop][1]}" if graph_type(g[prop]) != props[prop][1] else f"{label}.{prop}: value differs from CSV", f"{key}={k}: sent {g[prop]!r}, expected {e[prop]!r}"))
    for label in set(node_rows) - set(NODE_SCHEMA):
        problems.append((f"label {label} is not in the schema", ""))

    # compare relationships
    exp_rels = Counter()
    for a, t, b, file, start_key, end_col in REL_SCHEMA:
        for r in csv_rows(file):
            exp_rels[(a, t, b, start_key(r), int(r[end_col]))] += 1
    missing, extra = exp_rels - rel_rows, rel_rows - exp_rels
    if missing or extra:
        problems.append((f"relationships: {sum(missing.values()):,} expected but not sent, "
                         f"{sum(extra.values()):,} unexpected",
                         f"missing e.g. {list(missing)[:2]}, unexpected e.g. {list(extra)[:2]}"))

    # dedupe the per-node messages into counts
    counts, first = Counter(m for m, _ in problems), {}
    for m, ex in problems:
        first.setdefault(m, ex)
    details = [f"{m} [{n:,}x]" + (f", e.g. {first[m]}" if first[m] else "") for m, n in counts.items()]
    n_nodes = sum(len(v) for v in node_rows.values())
    n_rels = sum(rel_rows.values())
    rep.check(f"[offline] loaders would create {n_nodes:,} nodes (expected {EXPECTED_NODES:,}) "
              f"and {n_rels:,} relationships (expected {EXPECTED_RELS:,})",
              n_nodes == EXPECTED_NODES and n_rels == EXPECTED_RELS)
    rep.check("[offline] every node row and relationship the loaders send matches the CSVs and the schema",
              not details, details)
    uses_merge = all(("MERGE" in c) for c, _ in sent if "CONSTRAINT" not in c)
    rep.check("[offline] every write uses MERGE (a second run cannot duplicate)", uses_merge)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="also run Create_kg.py again and check nothing changed")
    ap.add_argument("--offline", action="store_true",
                    help="no database: check what the loaders would send against the CSVs")
    ap.add_argument("--config", default=None, help="path to config.txt (default: next to this script)")
    args = ap.parse_args()

    if args.offline:
        rep = Report()
        offline(rep)
        print(f"\n{'all checks passed' if not rep.failed else f'{len(rep.failed)} checks FAILED'}")
        sys.exit(1 if rep.failed else 0)

    cfg = read_config(args.config)
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    rep = Report()
    try:
        driver.verify_connectivity()
        with driver.session(database=cfg.get("DATABASE") or None) as session:
            check_totals_and_breakdown(session, rep)
            check_relationship_targets(session, rep)
            check_properties(session, rep)
            check_cypher_types(session, rep)
            check_constraints(session, rep)
            if args.rerun:
                check_rerun(session, rep)
    finally:
        driver.close()

    print()
    if rep.failed:
        print(f"{len(rep.failed)} of {len(rep.results)} checks FAILED")
        if not args.rerun:
            print("(the second-run check was not done: use --rerun)")
        sys.exit(1)
    print(f"all {len(rep.results)} checks passed" + ("" if args.rerun else
          " (second-run check not done: run again with --rerun)"))


if __name__ == "__main__":
    main()
