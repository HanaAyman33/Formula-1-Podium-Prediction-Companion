import csv
from datetime import date
from neo4j import GraphDatabase


def read_config(path="config.txt"):
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


def load_entities(session):
    pass


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