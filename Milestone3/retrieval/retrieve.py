"""
Milestone 3 - Task B: retrieve(intent, entities).

    from retrieve import retrieve
    out = retrieve("driver_stats", entities, message="How many podiums did Leclerc get in 2022?")

Returns a dict:
    status      "ok" | "empty" | "missing_entities" | "unsupported_intent" | "error"
    empty       True when the graph returned nothing  -> the app shows NO_DATA_MESSAGE, the LLM never fills the gap
    message     NO_DATA_MESSAGE when empty, else a short reason / None
    template    "intent.name" of the template that ran
    cypher      the parameterised Cypher that ran
    params      the parameters passed with it
    rows        list of dict rows from the template
    graph       {"kind": "graph", "nodes": [...], "edges": [...], "statements": [...]}   for list-type answers
                {"kind": "value", "column": ..., "value": ...}                           for plain numbers
                {"kind": "value", "column": None, "value": {row}}                        for one-row multi-number answers
                {"kind": "table"}                                                        for other lists (rows hold the answer)
                None when nothing ran

`entities` can be Member 1's flat to_params() dict (driver_id, driver2_id, constructor_id, circuit_id,
race_id, season, year_from, year_to) or the raw extraction dict (drivers, constructors, circuit, race{...}).
Optional extras: race_name + race_year (or race{name, year}), country, year_a / year_b.

Which of an intent's 4 templates runs is decided by (1) keywords in `message` if given, then (2) which entities
are present. Pass template="intent.name" to force one.
"""
import csv
import os
import re

from cypher_templates import TEMPLATES

NO_DATA_MESSAGE = "The knowledge graph has no data for that question."
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ small lookups from the CSVs (no DB call needed)
def _read(name):
    with open(os.path.join(HERE, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


_RACES = None
_CIRCUITS = None


def _races():
    global _RACES
    if _RACES is None:
        _RACES = {int(r["raceId"]): {"name": r["name"], "year": int(r["year"]), "circuit_id": int(r["circuitId"])}
                  for r in _read("races.csv")}
    return _RACES


def _circuits():
    global _CIRCUITS
    if _CIRCUITS is None:
        _CIRCUITS = {int(r["circuitId"]): r["country"] for r in _read("circuits.csv")}
    return _CIRCUITS


# ------------------------------------------------------------------ entity normalisation
def normalise(entities):
    """Flat dict with the keys retrieve() needs, from either to_params() output or the raw extraction dict."""
    e = dict(entities or {})
    drivers = e.get("drivers") or []
    cons = e.get("constructors") or []
    race = e.get("race") if isinstance(e.get("race"), dict) else {}
    circuit = e.get("circuit") if isinstance(e.get("circuit"), dict) else {}

    n = {
        "driver_id": e.get("driver_id") or (drivers[0].get("driver_id") if drivers else None),
        "driver2_id": e.get("driver2_id") or (drivers[1].get("driver_id") if len(drivers) > 1 else None),
        "constructor_id": e.get("constructor_id") or (cons[0].get("constructor_id") if cons else None),
        "circuit_id": e.get("circuit_id") or circuit.get("circuit_id") or race.get("circuit_id"),
        "season": e.get("season"),
        "year_from": e.get("year_from"),
        "year_to": e.get("year_to"),
        "country": e.get("country"),
        "year_a": e.get("year_a"),
        "year_b": e.get("year_b"),
    }
    race_id = e.get("race_id") or race.get("race_id")
    name = e.get("race_name") or race.get("name")
    year = e.get("race_year") or race.get("year")
    if race_id and int(race_id) in _races():
        info = _races()[int(race_id)]
        name, year = name or info["name"], year or info["year"]
        n["circuit_id"] = n["circuit_id"] or info["circuit_id"]
    n["race_name"], n["race_year"] = name, year
    if n["circuit_id"] and not n["country"]:
        n["country"] = _circuits().get(int(n["circuit_id"]))
    return n


# ------------------------------------------------------------------ choosing the template
def _has(msg, *words):
    return any(re.search(rf"\b{w}", msg) for w in words)


def choose_template(intent, n, message=""):
    m = (message or "").lower()
    race = bool(n["race_name"] and n["race_year"])
    year = n["season"] or n["race_year"]

    if intent == "driver_stats":
        if n["circuit_id"]:
            return "driver_stats.avg_finish_at_circuit"
        if n["season"]:
            return "driver_stats.season_wins_podiums" if _has(m, "podium", "win", "won") else "driver_stats.season_points"
        return "driver_stats.career_podiums"

    if intent == "constructor_stats":
        if _has(m, "reliab", "retire", "dnf", "breakdown") or (n["year_from"] and n["year_to"] and not n["season"]):
            return "constructor_stats.reliability"
        if _has(m, "driver", "drove", "who "):
            return "constructor_stats.season_drivers"
        if _has(m, "podium", "win", "won"):
            return "constructor_stats.season_wins_podiums"
        return "constructor_stats.season_points"

    if intent == "head_to_head":
        if n["circuit_id"]:
            return "head_to_head.at_circuit"
        if _has(m, "qualif", "pole", "grid"):
            return "head_to_head.qualifying_battle"
        if _has(m, "point", "score"):
            return "head_to_head.points_comparison"
        return "head_to_head.finish_ahead"

    if intent == "race_result":
        if _has(m, "fastest", "quickest"):
            return "race_result.fastest_lap"
        if _has(m, "retire", "dnf", "crash", "why", "out of"):
            return "race_result.retirements"
        if race:
            return "race_result.podium"
        return "race_result.winner_by_circuit_year"

    if intent == "circuit_info":
        if _has(m, "win", "won"):
            return "circuit_info.most_wins"
        if _has(m, "how many", "held", "hosted"):
            return "circuit_info.races_held"
        if _has(m, "which circuits", "circuits in", "tracks in") and n["country"]:
            return "circuit_info.circuits_in_country"
        return "circuit_info.location"

    if intent == "qualifying":
        if _has(m, "gain", "grid", "places"):
            return "qualifying.grid_gains"
        if race and not n["driver_id"]:
            return "qualifying.pole_sitter"
        if _has(m, "q3"):
            return "qualifying.q3_appearances"
        if n["driver_id"] and year and not _has(m, "pole"):
            return "qualifying.q3_appearances"
        if n["driver_id"]:
            return "qualifying.pole_count"
        return "qualifying.pole_sitter"

    if intent == "pit_stops":
        if race:
            return "pit_stops.stops_per_driver" if _has(m, "how many", "number of", "stops per", "each driver") \
                else "pit_stops.fastest_in_race"
        if n["driver_id"] and year:
            return "pit_stops.driver_avg_season"
        return "pit_stops.fastest_in_season"

    if intent == "season_history":
        if _has(m, "change", "switch", "moved", "left"):
            return "season_history.team_changes"
        if _has(m, "how many drivers", "competed", "entered", "grid size"):
            return "season_history.drivers_in_season"
        if _has(m, "calendar", "grands prix", "races were", "which races", "schedule"):
            return "season_history.calendar"
        if n["year_from"] and n["year_to"] and not n["season"]:
            return "season_history.team_changes"
        return "season_history.points_leader"
    return None


# ------------------------------------------------------------------ parameters per template
def build_params(key, n):
    """(params, missing) for one template. Only values the template's Cypher uses are returned."""
    year = n["season"] or n["race_year"]
    ya = n["year_a"] or n["year_from"] or ((n["season"] - 1) if n["season"] else None)
    yb = n["year_b"] or n["year_to"] or n["season"]
    full = {
        "driverId": n["driver_id"], "driverAId": n["driver_id"], "driverBId": n["driver2_id"],
        "constructorId": n["constructor_id"], "circuitId": n["circuit_id"],
        "raceName": n["race_name"], "year": year, "yearFrom": n["year_from"], "yearTo": n["year_to"],
        "country": n["country"], "yearA": ya, "yearB": yb,
    }
    if key == "constructor_stats.reliability":          # range required; a single season counts as a range
        full["yearFrom"] = n["year_from"] or year
        full["yearTo"] = n["year_to"] or year
    if key in ("qualifying.grid_gains",) and n["race_year"] and not n["season"]:
        full["year"] = n["race_year"]

    cypher = TEMPLATES[key]["cypher"]
    used = {p for p in re.findall(r"\$(\w+)", cypher)}
    params = {p: full.get(p) for p in used}
    required = {p for p in used if p not in ("yearFrom", "yearTo", "constructorId")}
    if key == "constructor_stats.reliability":
        required |= {"yearFrom", "yearTo"}
    missing = sorted(p for p in required if params[p] is None)
    return params, missing


# ------------------------------------------------------------------ graph output
RES = "MATCH (res)-[x:FOR_RACE|BY_DRIVER|FOR_CONSTRUCTOR|HAS_STATUS]->(m) RETURN res AS s, type(x) AS rel, m AS t"
RACE = "MATCH (r)-[x:AT_CIRCUIT|IN_SEASON]->(m) RETURN r AS s, type(x) AS rel, m AS t"

# Each entry: list of parameterised queries returning (s, rel, t). Same filters/order/limit as the template.
GRAPH_QUERIES = {
    "race_result.podium": ["""
MATCH (res:Result)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}) WHERE res.position <= 3
WITH res ORDER BY res.position LIMIT 3
""" + RES],
    "race_result.winner_by_circuit_year": ["""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(ci:Circuit {circuitId: $circuitId}), (r)-[:IN_SEASON]->(s:Season {year: $year})
WHERE res.position = 1
WITH res ORDER BY r.date LIMIT 5
""" + RES],
    "race_result.fastest_lap": ["""
MATCH (res:Result)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}) WHERE res.rank = 1
WITH res LIMIT 1
""" + RES],
    "race_result.retirements": ["""
MATCH (res:Result)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}), (res)-[:HAS_STATUS]->(st:Status)
WHERE NOT (st.status = 'Finished' OR st.status STARTS WITH '+')
WITH res ORDER BY res.laps DESC LIMIT 25
""" + RES],
    "qualifying.pole_sitter": ["""
MATCH (res:Qualifying)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}) WHERE res.position = 1
WITH res LIMIT 1
""" + RES],
    "qualifying.grid_gains": ["""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(ci:Circuit {circuitId: $circuitId}), (r)-[:IN_SEASON]->(s:Season {year: $year})
WHERE res.grid > 0 AND res.position IS NOT NULL
WITH res ORDER BY res.grid - res.position DESC LIMIT 5
""" + RES],
    "pit_stops.fastest_in_race": ["""
MATCH (p:PitStop)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}) WHERE p.milliseconds IS NOT NULL
WITH p ORDER BY p.milliseconds ASC LIMIT 3
MATCH (p)-[x:FOR_RACE|BY_DRIVER]->(m) RETURN p AS s, type(x) AS rel, m AS t"""],
    "season_history.calendar": ["""
MATCH (r:Race)-[:IN_SEASON]->(s:Season {year: $year})
WITH r ORDER BY r.round LIMIT 30
""" + RACE],
}

# plain-number templates: the single column that is "the answer"
HEADLINE = {
    "driver_stats.season_points": "points", "driver_stats.season_wins_podiums": "podiums",
    "driver_stats.career_podiums": "podiums", "driver_stats.avg_finish_at_circuit": "avg_classified_finish",
    "constructor_stats.season_points": "points", "constructor_stats.season_wins_podiums": "races_won",
    "constructor_stats.reliability": "finish_rate_pct", "circuit_info.races_held": "races_held",
    "qualifying.pole_count": "poles", "qualifying.q3_appearances": "q3_appearances",
    "pit_stops.driver_avg_season": "avg_seconds", "season_history.drivers_in_season": "drivers",
}

# label -> (id property, display function)
NODE_KEY = {"Driver": "driverId", "Constructor": "constructorId", "Circuit": "circuitId", "Race": "raceId",
            "Season": "year", "Status": "statusId", "Result": "resultId", "Qualifying": "qualifyId",
            "PitStop": "eventId"}


def _label(node):
    labels = list(node.labels)
    return labels[0] if labels else "Node"


def _plain(v):
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)   # neo4j dates etc.


def _display(label, p):
    if label == "Driver":
        return f"{p.get('forename', '')} {p.get('surname', '')}".strip()
    if label == "Constructor":
        return p.get("constructor_name")
    if label == "Circuit":
        return p.get("circuit_name")
    if label == "Race":
        return f"{p.get('year')} {p.get('name')}"
    if label == "Season":
        return str(p.get("year"))
    if label == "Status":
        return p.get("status")
    if label == "Result":
        return f"Result #{p.get('resultId')} (P{p.get('position')})"
    if label == "Qualifying":
        return f"Qualifying #{p.get('qualifyId')} (P{p.get('position')})"
    if label == "PitStop":
        return f"Pit stop {p.get('stop')} lap {p.get('lap')} ({p.get('duration')})"
    return label


def build_graph(session, key, params):
    """Run the graph queries for this template and turn the triples into nodes/edges/statements."""
    nodes, edges, seen_e, statements = {}, [], set(), []

    def add(node):
        label = _label(node)
        props = {k: _plain(v) for k, v in dict(node).items()}
        nid = f"{label}:{props.get(NODE_KEY.get(label))}"
        nodes.setdefault(nid, {"id": nid, "label": label, "name": _display(label, props), "properties": props})
        return nid, label

    for q in GRAPH_QUERIES[key]:
        for rec in session.run(q, params):
            s, rel, t = rec["s"], rec["rel"], rec["t"]
            sid, sl = add(s)
            tid, tl = add(t)
            if (sid, rel, tid) not in seen_e:
                seen_e.add((sid, rel, tid))
                edges.append({"source": sid, "target": tid, "type": rel})
                stmt = f"({sl})-[:{rel}]->({tl})"
                if stmt not in statements:
                    statements.append(stmt)
    return {"kind": "graph", "nodes": list(nodes.values()), "edges": edges, "statements": statements}


# ------------------------------------------------------------------ main entry point
def _all_empty(rows):
    return not rows or all(all(v is None for v in r.values()) for r in rows)


def _result(status, **kw):
    base = {"status": status, "empty": False, "message": None, "template": None, "cypher": None,
            "params": None, "rows": [], "graph": None}
    base.update(kw)
    return base


def retrieve(intent, entities, message="", session=None, template=None):
    """Pick the template for (intent, entities), run it, and return {cypher, params, rows, graph, ...}.

    session: an open neo4j session. If None, one is opened from config.txt and closed afterwards.
    """
    n = normalise(entities)
    key = template or choose_template(intent, n, message)
    if key not in TEMPLATES:
        return _result("unsupported_intent", message=f"No graph template for intent '{intent}'.")

    params, missing = build_params(key, n)
    if missing:
        return _result("missing_entities", template=key, params=params,
                       message="Missing information: " + ", ".join(missing))
    cypher = TEMPLATES[key]["cypher"]

    own = session is None
    driver = None
    try:
        if own:
            from neo4j import GraphDatabase
            from Create_kg import read_config
            cfg = read_config()
            driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
            session = driver.session(database=cfg.get("DATABASE") or None)

        rows = [dict(r) for r in session.run(cypher, params)]
        if _all_empty(rows):
            return _result("empty", empty=True, message=NO_DATA_MESSAGE, template=key,
                           cypher=cypher, params=params, rows=[], graph=None)

        if key in GRAPH_QUERIES:
            graph = build_graph(session, key, params)
        elif key in HEADLINE and len(rows) == 1:
            graph = {"kind": "value", "column": HEADLINE[key], "value": rows[0].get(HEADLINE[key])}
        elif len(rows) == 1:
            graph = {"kind": "value", "column": None, "value": rows[0]}
        else:
            graph = {"kind": "table"}
        return _result("ok", template=key, cypher=cypher, params=params, rows=rows, graph=graph)
    except Exception as exc:                                    # connection / syntax errors: never invent an answer
        return _result("error", template=key, cypher=cypher, params=params,
                       message=f"Graph query failed: {exc}")
    finally:
        if own and session is not None:
            session.close()
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    import json
    import sys
    demo = [
        ("driver_stats", {"driver_id": 844, "season": 2022}, "How many podiums did Leclerc get in 2022?"),
        ("race_result", {"race_name": "German Grand Prix", "race_year": 2019}, "Who was on the podium?"),
        ("circuit_info", {"circuit_id": 14}, "How many Grands Prix were held at Monza?"),
        ("driver_stats", {"driver_id": 844, "season": 1900}, "How many points did Leclerc score in 1900?"),
    ]
    for intent, ents, msg in demo:
        out = retrieve(intent, ents, msg)
        print(f"\n### {intent}: {msg}\nstatus={out['status']} template={out['template']}")
        print("params:", out["params"])
        print("rows:", out["rows"][:3])
        g = out["graph"]
        print("graph:", json.dumps(g, default=str)[:600] if g else g, "| message:", out["message"])
    sys.exit(0)
