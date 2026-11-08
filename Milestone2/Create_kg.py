import os
import csv
from datetime import date
from neo4j import GraphDatabase


def csv_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def read_config(path=None):
    path = path or csv_path("config.txt")
    cfg = {}
    with open(path, encoding="utf-8") as f:
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


def clean(d):
    return {k: v for k, v in d.items() if v is not None}


def run_batches(session, query, rows, size=1000):
    for i in range(0, len(rows), size):
        session.run(query, rows=rows[i:i + size])


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


def load_entities(session):
    load_seasons(session)
    load_status(session)
    load_circuits(session)


def load_results(session):
    pass


def load_qualifying(session):
    pass


def load_pitstops(session):
    pass


def main():
    cfg = read_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    with driver.session(database=cfg.get("DATABASE")) as session:
        for label, prop in CONSTRAINTS:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
        load_entities(session)
        load_results(session)
        load_qualifying(session)
        load_pitstops(session)
    driver.close()


if __name__ == "__main__":
    main()