import os
import csv
from datetime import date
from neo4j import GraphDatabase


def csv_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def read_config(path=None):
    path = path or csv_path("config.txt")
    cfg = {}
    # utf-8-sig drops the BOM that Windows Notepad adds, so the first key stays "URI"
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def is_missing(v):
    return v is None or v == "" or v == r"\N"


def to_int(v):
    return None if is_missing(v) else int(float(v))


def to_float(v):
    return None if is_missing(v) else float(v)


def to_date(v):
    return None if is_missing(v) else date.fromisoformat(v)


def to_str(v):
    # for text and time columns (q1, duration, fastestLapTime, ...): kept as strings, missing -> omitted
    return None if is_missing(v) else v


def read_rows(name):
    with open(csv_path(name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean(d):
    return {k: v for k, v in d.items() if v is not None}


def run_batches(session, query, rows, size=1000):
    for i in range(0, len(rows), size):
        # consume() waits for the batch to finish, so any Cypher error is raised here
        session.run(query, rows=rows[i:i + size]).consume()


CONSTRAINTS = [
    ("Driver", "driverId"), ("Constructor", "constructorId"),
    ("Circuit", "circuitId"), ("Season", "year"),
    ("Status", "statusId"), ("Race", "raceId"),
    ("Result", "resultId"), ("Qualifying", "qualifyId"),
    ("PitStop", "eventId"),
]


def load_seasons(session):
    with open(csv_path("seasons.csv"), encoding="utf-8") as f:
        rows = [{"year": to_int(r["year"])} for r in csv.DictReader(f)]
    run_batches(session, "UNWIND $rows AS row MERGE (s:Season {year: row.year})", rows)


def load_status(session):
    with open(csv_path("status.csv"), encoding="utf-8") as f:
        rows = [{"statusId": to_int(r["statusId"]), "status": r["status"]}
                for r in csv.DictReader(f)]
    run_batches(session,
                "UNWIND $rows AS row MERGE (s:Status {statusId: row.statusId}) SET s.status = row.status",
                rows)


def load_circuits(session):
    with open(csv_path("circuits.csv"), encoding="utf-8") as f:
        rows = [clean({
            "circuitId": to_int(r["circuitId"]),
            "circuitRef": r["circuitRef"],
            "circuit_name": r["name"],
            "circuit_loc": r["location"],
            "circuit_country": r["country"],
            "lat": to_float(r["lat"]),
            "lng": to_float(r["lng"]),
            "alt": to_int(r["alt"]),
        }) for r in csv.DictReader(f)]
    run_batches(session,
                "UNWIND $rows AS row MERGE (c:Circuit {circuitId: row.circuitId}) SET c += row",
                rows)

def load_constructors(session):
    with open(csv_path("constructors.csv"), encoding="utf-8") as f:
        rows = [clean({
            "constructorId": to_int(r["constructorId"]),
            "constructorRef": r["constructorRef"],
            "constructor_name": r["name"],
            "constructor_nationality": r["nationality"],
        }) for r in csv.DictReader(f)]
    run_batches(session,
                "UNWIND $rows AS row MERGE (c:Constructor {constructorId: row.constructorId}) SET c += row",
                rows)

def load_drivers(session):
    with open(csv_path("drivers.csv"), encoding="utf-8") as f:
        rows = [clean({
            "driverId": to_int(r["driverId"]),
            "driverRef": r["driverRef"],
            "number": to_int(r["number"]),
            "code": to_str(r["code"]),
            "forename": r["forename"],
            "surname": r["surname"],
            "dob": to_date(r["dob"]),
            "nationality": r["nationality"],
        }) for r in csv.DictReader(f)]
    run_batches(session,
                "UNWIND $rows AS row MERGE (d:Driver {driverId: row.driverId}) SET d += row",
                rows)

def load_races(session):
    with open(csv_path("races.csv"), encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    nodes = [clean({
        "raceId": to_int(r["raceId"]),
        "year": to_int(r["year"]),
        "round": to_int(r["round"]),
        "name": r["name"],
        "date": to_date(r["date"]),
        "time": to_str(r["time"]),
    }) for r in raw]
    run_batches(session,
                "UNWIND $rows AS row MERGE (r:Race {raceId: row.raceId}) SET r += row",
                nodes)

    links = [{"raceId": to_int(r["raceId"]),
              "circuitId": to_int(r["circuitId"]),
              "year": to_int(r["year"])} for r in raw]
    run_batches(session,
                "UNWIND $rows AS row "
                "MATCH (r:Race {raceId: row.raceId}), (c:Circuit {circuitId: row.circuitId}) "
                "MERGE (r)-[:AT_CIRCUIT]->(c)",
                links)
    run_batches(session,
                "UNWIND $rows AS row "
                "MATCH (r:Race {raceId: row.raceId}), (s:Season {year: row.year}) "
                "MERGE (r)-[:IN_SEASON]->(s)",
                links)

def load_entities(session):
    load_seasons(session)
    load_status(session)
    load_circuits(session)
    load_constructors(session)
    load_drivers(session)
    load_races(session)

def load_results(session):
    raw = read_rows("results.csv")

    # 1) Result fact nodes: one per row, missing values (\N) are left out by clean()
    nodes = [clean({
        "resultId": to_int(r["resultId"]),
        "number": to_int(r["number"]),
        "grid": to_int(r["grid"]),
        "position": to_int(r["position"]),
        "positionText": to_str(r["positionText"]),
        "positionOrder": to_int(r["positionOrder"]),
        "points": to_float(r["points"]),
        "laps": to_int(r["laps"]),
        "time": to_str(r["time"]),
        "milliseconds": to_int(r["milliseconds"]),
        "fastestLap": to_int(r["fastestLap"]),
        "rank": to_int(r["rank"]),
        "fastestLapTime": to_str(r["fastestLapTime"]),
        "fastestLapSpeed": to_float(r["fastestLapSpeed"]),
    }) for r in raw]
    run_batches(session,
                "UNWIND $rows AS row MERGE (res:Result {resultId: row.resultId}) SET res += row",
                nodes)

    # 2) the four links; the foreign keys are only used to find the parents, not stored on the node
    links = [{"resultId": to_int(r["resultId"]),
              "raceId": to_int(r["raceId"]),
              "driverId": to_int(r["driverId"]),
              "constructorId": to_int(r["constructorId"]),
              "statusId": to_int(r["statusId"])} for r in raw]
    run_batches(session,
                "UNWIND $rows AS row "
                "MATCH (res:Result {resultId: row.resultId}) "
                "MATCH (r:Race {raceId: row.raceId}) "
                "MATCH (d:Driver {driverId: row.driverId}) "
                "MATCH (c:Constructor {constructorId: row.constructorId}) "
                "MATCH (s:Status {statusId: row.statusId}) "
                "MERGE (res)-[:FOR_RACE]->(r) "
                "MERGE (res)-[:BY_DRIVER]->(d) "
                "MERGE (res)-[:FOR_CONSTRUCTOR]->(c) "
                "MERGE (res)-[:HAS_STATUS]->(s)",
                links)


def load_qualifying(session):
    pass


def load_pitstops(session):
    pass


def main():
    cfg = read_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    try:
        driver.verify_connectivity()
        # DATABASE is optional: grading's config.txt has it, ours may not (None = server default)
        with driver.session(database=cfg.get("DATABASE") or None) as session:
            for label, prop in CONSTRAINTS:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                            f"REQUIRE n.{prop} IS UNIQUE").consume()
            # parents first, then the facts that link to them
            load_entities(session)
            load_results(session)
            load_qualifying(session)
            load_pitstops(session)
    finally:
        driver.close()


if __name__ == "__main__":
    main()