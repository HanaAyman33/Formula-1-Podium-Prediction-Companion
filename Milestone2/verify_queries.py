"""Member 4: does the graph answer questions correctly, including ones nobody designed it around?

For every query below there is a Cypher version (run on Neo4j) and an independent pandas version
(run on the 9 CSV files). The two must return the same columns, the same rows and the same order.

  Part A: the 5 submitted queries, read from queries.txt
  Part B: 11 unseen-style queries (time filters, dates, missing values, joins the 5 never use)

If Cypher and pandas agree on questions the graph was not built for, the graph generalizes.

Usage:
  python verify_queries.py             # needs Neo4j (config.txt) and the built graph
  python verify_queries.py --expected  # no database: print the pandas answers only
"""
import argparse
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------------
# CSV side
# --------------------------------------------------------------------------------------------
def load_csvs():
    rd = lambda n: pd.read_csv(os.path.join(HERE, n), na_values=[r"\N"], keep_default_na=False)
    t = {n: rd(f"{n}.csv") for n in ["races", "results", "drivers", "constructors", "status",
                                     "circuits", "qualifying", "pit_stops", "seasons"]}
    t["qualifying"][["q1", "q2", "q3"]] = t["qualifying"][["q1", "q2", "q3"]].replace("", pd.NA)
    t["drivers"]["full"] = t["drivers"].forename + " " + t["drivers"].surname
    return t


def half_up(x, nd):
    """Cypher's round() rounds halves up; Python's round() does not."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def frame(df, cols):
    return list(cols), [tuple(r) for r in df[list(cols)].itertuples(index=False)]


def results_full(t):
    return (t["results"].merge(t["races"][["raceId", "year", "round", "name", "date", "circuitId"]], on="raceId")
            .merge(t["drivers"][["driverId", "full", "driverRef", "dob"]], on="driverId")
            .merge(t["constructors"][["constructorId", "name"]].rename(columns={"name": "cname"}), on="constructorId")
            .merge(t["status"], on="statusId"))


# ---- Part A references ---------------------------------------------------------------------
def ref_q1(t):
    r = results_full(t).query("year == 2021")
    g = r.groupby(["driverId", "full"]).agg(Points=("points", "sum"),
                                             Wins=("position", lambda s: int((s == 1).sum()))).reset_index()
    g["Driver"] = g.full
    g = g.sort_values(["Points", "Wins", "Driver"], ascending=[False, False, True])
    return frame(g, ["Driver", "Points", "Wins"])


def ref_q2(t):
    r = results_full(t).query("2014 <= year <= 2021")
    r["cls"] = (r.status == "Finished") | r.status.str.startswith("+")
    g = r.groupby(["constructorId", "cname"]).agg(Entries=("resultId", "size"), Classified=("cls", "sum")).reset_index()
    g = g[g.Entries >= 100].copy()
    g["Classified"] = g.Classified.astype(int)
    g["Reliability"] = g.Classified / g.Entries
    g["Constructor"] = g.cname
    g = g.sort_values(["Reliability", "Constructor"], ascending=[False, True])
    return frame(g, ["Constructor", "Entries", "Classified", "Reliability"])


def ref_q3(t):
    r = results_full(t)
    h = r[r.driverId == 1].groupby(["raceId", "constructorId", "cname"]).positionOrder.min().rename("h")
    b = r[r.driverId == 822].groupby(["raceId", "constructorId", "cname"]).positionOrder.min().rename("b")
    j = pd.concat([h, b], axis=1, join="inner").reset_index()
    g = j.groupby("cname").apply(lambda d: pd.Series({
        "Shared_Races": len(d), "Hamilton_Ahead": int((d.h < d.b).sum()),
        "Bottas_Ahead": int((d.b < d.h).sum())}), include_groups=False).reset_index()
    g["Constructor"] = g.cname
    return frame(g.sort_values("Constructor"), ["Constructor", "Shared_Races", "Hamilton_Ahead", "Bottas_Ahead"])


def ref_q4(t):
    r = results_full(t)
    r = r[(r.circuitId == 6) & (r.date < "2020-01-01")
          & ~r.status.isin(["Did not qualify", "Did not prequalify", "Withdrew"])]
    g = r.groupby(["driverId", "full"]).agg(Starts=("raceId", "nunique"), avg=("positionOrder", "mean"),
                                             Wins=("position", lambda s: int((s == 1).sum()))).reset_index()
    g = g[g.Starts >= 5].copy()
    g["Avg_Finish"] = g.avg.map(lambda x: half_up(x, 4))
    g["Driver"] = g.full
    g = g.sort_values(["Avg_Finish", "Wins", "Driver"], ascending=[True, False, True])
    return frame(g, ["Driver", "Starts", "Avg_Finish", "Wins"])


def ref_q5(t):
    r = results_full(t)

    def primary(year):
        # primary constructor = most race entries (Result nodes) that season; a tie goes to the lower constructorId
        g = r[r.year == year].groupby(["driverId", "constructorId"]).size().rename("n").reset_index()
        g = g.sort_values(["driverId", "n", "constructorId"], ascending=[True, False, True])
        return g.drop_duplicates("driverId").set_index("driverId").constructorId

    j = pd.concat([primary(2020).rename("c20"), primary(2021).rename("c21")], axis=1, join="inner")
    names = sorted(t["drivers"].set_index("driverId").full[j.index[j.c20 != j.c21]])
    return ["Drivers_Changed", "Drivers"], [(len(names), names)]


# ---- Part B: unseen-style queries ----------------------------------------------------------
def ref_wins_2008(t):
    r = results_full(t).query("year == 2008 and position == 1")
    g = r.groupby("cname").size().rename("Wins").reset_index().rename(columns={"cname": "Constructor"})
    return frame(g.sort_values(["Wins", "Constructor"], ascending=[False, True]), ["Constructor", "Wins"])


def ref_pitstops_2023(t):
    p = t["pit_stops"].merge(t["races"], on="raceId").query("year == 2023")
    g = p.groupby(["round", "name"]).size().rename("PitStops").reset_index()
    g = g.rename(columns={"round": "Round", "name": "Race"})
    return frame(g.sort_values("Round"), ["Round", "Race", "PitStops"])


def ref_born_after_1995(t):
    d = t["drivers"][t["drivers"].dob > "1995-12-31"].copy()
    d["Driver"], d["DOB"] = d.full, d.dob
    return frame(d.sort_values(["DOB", "Driver"]), ["Driver", "DOB"])


def ref_fastest_monza(t):
    r = results_full(t)
    r = r[(r.circuitId == 14) & r.fastestLapTime.notna()].copy()
    r["Driver"], r["Year"], r["LapTime"] = r.full, r.year, r.fastestLapTime
    return frame(r.sort_values(["LapTime", "Year", "Driver"]).head(5), ["Driver", "Year", "LapTime"])


def ref_q3_abu_dhabi_2021(t):
    q = t["qualifying"].merge(t["races"], on="raceId").merge(t["drivers"], on="driverId")
    q = q[(q.year == 2021) & (q.name == "Abu Dhabi Grand Prix") & q.q3.notna()].copy()
    q["Position"], q["Driver"], q["Q3"] = q.position, q.full, q.q3
    return frame(q.sort_values("Position"), ["Position", "Driver", "Q3"])


def ref_age_first_race(t):
    r = results_full(t)
    r = r[r.driverRef.isin(["hamilton", "max_verstappen", "leclerc", "norris"])]
    g = r.groupby(["driverRef", "dob"]).date.min().reset_index()
    def years(dob, d):
        a, b = pd.Timestamp(dob), pd.Timestamp(d)
        return b.year - a.year - ((b.month, b.day) < (a.month, a.day))
    g["DriverRef"], g["FirstRace"] = g.driverRef, g.date
    g["AgeYears"] = [years(a, b) for a, b in zip(g.dob, g.date)]
    return frame(g.sort_values("DriverRef"), ["DriverRef", "FirstRace", "AgeYears"])


def ref_races_per_season(t):
    r = t["races"].query("2015 <= year <= 2024")
    g = r.groupby("year").size().rename("Races").reset_index().rename(columns={"year": "Season"})
    g["YearMatches"] = g.Races
    return frame(g.sort_values("Season"), ["Season", "Races", "YearMatches"])


def ref_races_before_2000(t):
    n = int((t["races"].date < "2000-01-01").sum())
    return ["Races"], [(n,)]


def ref_unclassified_2021(t):
    r = results_full(t).query("year == 2021")
    return ["Unclassified"], [(int(r.position.isna().sum()),)]


def ref_pit_ms_by_constructor_2021(t):
    p = t["pit_stops"].merge(t["races"][["raceId", "year"]], on="raceId").query("year == 2021")
    p = p.merge(t["results"][["raceId", "driverId", "constructorId"]], on=["raceId", "driverId"])
    p = p.merge(t["constructors"][["constructorId", "name"]], on="constructorId")
    g = p.groupby("name").agg(Stops=("stop", "size"), AvgMs=("milliseconds", "mean")).reset_index()
    g = g.rename(columns={"name": "Constructor"})
    return frame(g.sort_values("Constructor"), ["Constructor", "Stops", "AvgMs"])


def ref_pole_to_win_2019(t):
    q = t["qualifying"].merge(t["races"], on="raceId").query("year == 2019 and position == 1")
    w = t["results"][t["results"].position == 1][["raceId", "driverId"]].rename(columns={"driverId": "winner"})
    j = q.merge(w, on="raceId")
    return ["Races", "PoleToWin"], [(len(j), int((j.driverId == j.winner).sum()))]


def ref_last_race_2023(t):
    r = t["races"].query("year == 2023").sort_values("date", ascending=False).head(1)
    return ["Race", "Date"], [(r.name.iloc[0], r.date.iloc[0])]


UNSEEN = [
    ("Wins per constructor, 2008", ref_wins_2008, """
MATCH (res:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(:Season {year: 2008})
MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WHERE res.position = 1
RETURN c.constructor_name AS Constructor, count(*) AS Wins
ORDER BY Wins DESC, Constructor ASC"""),
    ("Pit stops per race, 2023", ref_pitstops_2023, """
MATCH (p:PitStop)-[:FOR_RACE]->(r:Race)
WHERE r.year = 2023
RETURN r.round AS Round, r.name AS Race, count(p) AS PitStops
ORDER BY Round ASC"""),
    ("Drivers born after 1995 (date comparison)", ref_born_after_1995, """
MATCH (d:Driver)
WHERE d.dob > date('1995-12-31')
RETURN d.forename + ' ' + d.surname AS Driver, d.dob AS DOB
ORDER BY DOB ASC, Driver ASC"""),
    ("Five fastest race laps at Monza", ref_fastest_monza, """
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(:Circuit {circuitRef: 'monza'})
MATCH (res)-[:BY_DRIVER]->(d:Driver)
WHERE res.fastestLapTime IS NOT NULL
RETURN d.forename + ' ' + d.surname AS Driver, r.year AS Year, res.fastestLapTime AS LapTime
ORDER BY LapTime ASC, Year ASC, Driver ASC
LIMIT 5"""),
    ("Q3 times, 2021 Abu Dhabi GP (q3 left out when empty)", ref_q3_abu_dhabi_2021, """
MATCH (q:Qualifying)-[:FOR_RACE]->(r:Race {year: 2021, name: 'Abu Dhabi Grand Prix'})
MATCH (q)-[:BY_DRIVER]->(d:Driver)
WHERE q.q3 IS NOT NULL
RETURN q.position AS Position, d.forename + ' ' + d.surname AS Driver, q.q3 AS Q3
ORDER BY Position ASC"""),
    ("Age at first race (duration.between on dates)", ref_age_first_race, """
MATCH (res:Result)-[:BY_DRIVER]->(d:Driver), (res)-[:FOR_RACE]->(r:Race)
WHERE d.driverRef IN ['hamilton', 'max_verstappen', 'leclerc', 'norris']
WITH d, min(r.date) AS FirstRace
RETURN d.driverRef AS DriverRef, FirstRace, duration.between(d.dob, FirstRace).years AS AgeYears
ORDER BY DriverRef ASC"""),
    ("Races per season via IN_SEASON, and Race.year agrees", ref_races_per_season, """
MATCH (r:Race)-[:IN_SEASON]->(s:Season)
WHERE s.year >= 2015 AND s.year <= 2024
RETURN s.year AS Season, count(r) AS Races, sum(CASE WHEN r.year = s.year THEN 1 ELSE 0 END) AS YearMatches
ORDER BY Season ASC"""),
    ("Races before 2000-01-01", ref_races_before_2000, """
MATCH (r:Race)
WHERE r.date < date('2000-01-01')
RETURN count(r) AS Races"""),
    ("Unclassified results in 2021 (position IS NULL)", ref_unclassified_2021, """
MATCH (res:Result)-[:FOR_RACE]->(:Race {year: 2021})
WHERE res.position IS NULL
RETURN count(res) AS Unclassified"""),
    ("Average pit-stop time per constructor, 2021 (PitStop joined to Result)", ref_pit_ms_by_constructor_2021, """
MATCH (p:PitStop)-[:FOR_RACE]->(r:Race {year: 2021})
MATCH (p)-[:BY_DRIVER]->(d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r)
MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
RETURN c.constructor_name AS Constructor, count(p) AS Stops, avg(p.milliseconds) AS AvgMs
ORDER BY Constructor ASC"""),
    ("Pole positions converted to wins, 2019", ref_pole_to_win_2019, """
MATCH (q:Qualifying)-[:FOR_RACE]->(r:Race)
WHERE r.year = 2019 AND q.position = 1
MATCH (q)-[:BY_DRIVER]->(pole:Driver)
MATCH (w:Result)-[:FOR_RACE]->(r)
WHERE w.position = 1
MATCH (w)-[:BY_DRIVER]->(winner:Driver)
RETURN count(r) AS Races, sum(CASE WHEN pole = winner THEN 1 ELSE 0 END) AS PoleToWin"""),
    ("Last race of 2023", ref_last_race_2023, """
MATCH (r:Race)-[:IN_SEASON]->(:Season {year: 2023})
RETURN r.name AS Race, r.date AS Date
ORDER BY Date DESC
LIMIT 1"""),
]

PART_A = {1: ref_q1, 2: ref_q2, 3: ref_q3, 4: ref_q4, 5: ref_q5}


# --------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------
def norm(v):
    if hasattr(v, "iso_format"):          # neo4j.time.Date
        return v.iso_format()
    if hasattr(v, "item"):                # numpy scalar
        v = v.item()
    return v


def same(a, b):
    a, b = norm(a), norm(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


def compare(name, cypher_cols, cypher_rows, exp_cols, exp_rows):
    problems = []
    if cypher_cols != exp_cols:
        problems.append(f"columns {cypher_cols} vs expected {exp_cols}")
    if len(cypher_rows) != len(exp_rows):
        problems.append(f"{len(cypher_rows)} rows vs expected {len(exp_rows)}")
    for i, (g, e) in enumerate(zip(cypher_rows, exp_rows)):
        if len(g) != len(e) or not all(same(x, y) for x, y in zip(g, e)):
            problems.append(f"row {i + 1}: got {tuple(norm(x) for x in g)}, expected {tuple(norm(x) for x in e)}")
            if len(problems) > 5:
                break
    print(f"[{'PASS' if not problems else 'FAIL'}] {name} ({len(exp_rows)} rows)")
    for p in problems[:6]:
        print(f"         {p}")
    return not problems


def read_queries_txt():
    path = os.path.join(HERE, "queries.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"^//(\d+)\. Query \d+[ \t]*$", text, flags=re.M)
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def print_expected(t):
    for n, fn in PART_A.items():
        cols, rows = fn(t)
        print(f"\n== Query {n} ({len(rows)} rows)\n{pd.DataFrame(rows, columns=cols).to_string(index=False)}")
    for name, fn, _ in UNSEEN:
        cols, rows = fn(t)
        print(f"\n== {name} ({len(rows)} rows)\n{pd.DataFrame(rows, columns=cols).head(30).to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expected", action="store_true", help="print the pandas answers only (no database)")
    args = ap.parse_args()
    t = load_csvs()
    if args.expected:
        print_expected(t)
        return

    sys.path.insert(0, HERE)
    from check import read_config
    from neo4j import GraphDatabase

    cfg = read_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    ok = []
    try:
        with driver.session(database=cfg.get("DATABASE") or None) as session:
            def run(cypher):
                res = session.run(cypher)
                rows = [tuple(r.values()) for r in res]
                return list(res.keys()), rows

            print("Part A: the 5 submitted queries (queries.txt) vs pandas")
            queries = read_queries_txt()
            for n, fn in PART_A.items():
                if n not in queries:
                    print(f"[FAIL] Query {n}: not found in queries.txt")
                    ok.append(False)
                    continue
                ok.append(compare(f"Query {n}", *run(queries[n]), *fn(t)))
            print("\nPart B: unseen-style queries vs pandas")
            for name, fn, cypher in UNSEEN:
                ok.append(compare(name, *run(cypher), *fn(t)))
    finally:
        driver.close()
    print(f"\n{sum(ok)} of {len(ok)} queries agree with pandas")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
