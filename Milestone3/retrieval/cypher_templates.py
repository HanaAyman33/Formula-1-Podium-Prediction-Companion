"""
Milestone 3 - Task A: parameterised Cypher templates (8 intents x 4 questions = 32 templates).

Rules followed by every template
  * only $parameters, never string-built Cypher
  * Milestone 2 directions: (Result)-[:BY_DRIVER|FOR_RACE|FOR_CONSTRUCTOR|HAS_STATUS]->(...),
    (Qualifying)-[:BY_DRIVER|FOR_RACE|FOR_CONSTRUCTOR]->(...), (PitStop)-[:FOR_RACE|BY_DRIVER]->(...),
    (Race)-[:AT_CIRCUIT]->(Circuit), (Race)-[:IN_SEASON]->(Season)
  * named output columns
  * each query ends with an aggregation or LIMIT

Entities are passed as IDs (driverId, constructorId, circuitId) resolved by entity extraction,
which also resolves ambiguous names, so no name matching happens inside Cypher.
Race names are the exact Race.name string, e.g. "German Grand Prix".

Usage
  python cypher_templates.py                 # run every template against Neo4j with sample params
  python cypher_templates.py driver_stats.avg_finish_at_circuit   # run one
  python cypher_templates.py --browser driver_stats.career_podiums  # print :param lines + query for Neo4j Browser
"""
import sys

DRIVER = "d.forename + ' ' + d.surname"

# statuses that are NOT a classified finish: everything except "Finished" and "+N Lap(s)"
FINISHED = "(st.status = 'Finished' OR st.status STARTS WITH '+')"
# retirement reasons that are driver/race incidents rather than the car failing
NON_MECHANICAL = ("['Accident','Collision','Spun off','Disqualified','Collision damage','Withdrew',"
                  "'Did not qualify','Did not prequalify','Excluded','Illness','Driver unwell','Injury',"
                  "'Fatal accident','Safety concerns','Safety','Not classified','107% Rule','Underweight']")

YEAR_RANGE = "($yearFrom IS NULL OR r.year >= $yearFrom) AND ($yearTo IS NULL OR r.year <= $yearTo)"

# Optional parameters default to None when the extractor did not find them.
OPTIONAL_DEFAULTS = {"yearFrom": None, "yearTo": None, "constructorId": None}

T = {}


def add(intent, name, entities, question, cypher, example):
    T[f"{intent}.{name}"] = {
        "intent": intent, "name": name, "entities": entities,
        "question": question, "cypher": cypher.strip(), "example": example,
    }


# ---------------------------------------------------------------- 1. driver_stats
add("driver_stats", "season_points", ["driver", "season"],
    "How many points did Verstappen score in 2022?",
    f"""
MATCH (d:Driver {{driverId: $driverId}})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}})
RETURN {DRIVER} AS driver, s.year AS season, sum(res.points) AS points, count(res) AS races_entered
""", {"driverId": 830, "year": 2022})

add("driver_stats", "season_wins_podiums", ["driver", "season"],
    "How many podiums did Leclerc get in 2022?",
    f"""
MATCH (d:Driver {{driverId: $driverId}})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}})
RETURN {DRIVER} AS driver, s.year AS season,
       sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
       sum(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS podiums,
       count(res) AS races_entered
""", {"driverId": 844, "year": 2022})

add("driver_stats", "career_podiums", ["driver", "(optional) year range"],
    "How many podiums did Hamilton have before 2020?",
    f"""
MATCH (d:Driver {{driverId: $driverId}})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)
WHERE {YEAR_RANGE}
RETURN {DRIVER} AS driver,
       sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
       sum(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS podiums,
       count(res) AS races_entered, min(r.year) AS first_season, max(r.year) AS last_season
""", {"driverId": 1, "yearFrom": None, "yearTo": 2019})

add("driver_stats", "avg_finish_at_circuit", ["driver", "circuit"],
    "What is Alonso's average finish at Monza?",
    f"""
MATCH (d:Driver {{driverId: $driverId}})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(c:Circuit {{circuitId: $circuitId}})
RETURN {DRIVER} AS driver, c.circuit_name AS circuit,
       round(avg(res.position), 2) AS avg_classified_finish,
       count(res) AS races_entered, count(res.position) AS classified_finishes,
       min(res.position) AS best_finish
""", {"driverId": 4, "circuitId": 14})

# ---------------------------------------------------------------- 2. constructor_stats
add("constructor_stats", "season_points", ["constructor", "season"],
    "How many points did Ferrari score in 2022?",
    """
MATCH (c:Constructor {constructorId: $constructorId})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {year: $year})
RETURN c.constructor_name AS constructor, s.year AS season, sum(res.points) AS points, count(DISTINCT r) AS races
""", {"constructorId": 6, "year": 2022})

add("constructor_stats", "season_wins_podiums", ["constructor", "season"],
    "How many races did Red Bull win in 2023?",
    """
MATCH (c:Constructor {constructorId: $constructorId})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {year: $year})
RETURN c.constructor_name AS constructor, s.year AS season,
       count(DISTINCT CASE WHEN res.position = 1 THEN r END) AS races_won,
       sum(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS podium_finishes,
       count(DISTINCT r) AS races
""", {"constructorId": 9, "year": 2023})

add("constructor_stats", "reliability", ["constructor", "year range"],
    "How reliable was Williams between 2014 and 2021?",
    f"""
MATCH (c:Constructor {{constructorId: $constructorId}})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:HAS_STATUS]->(st:Status),
      (res)-[:FOR_RACE]->(r:Race)
WHERE r.year >= $yearFrom AND r.year <= $yearTo
WITH c, res, st, {FINISHED} AS finished
RETURN c.constructor_name AS constructor, count(res) AS entries,
       sum(CASE WHEN finished THEN 1 ELSE 0 END) AS classified_finishes,
       sum(CASE WHEN NOT finished THEN 1 ELSE 0 END) AS retirements,
       sum(CASE WHEN NOT finished AND NOT st.status IN {NON_MECHANICAL} THEN 1 ELSE 0 END) AS mechanical_retirements,
       round(100.0 * sum(CASE WHEN finished THEN 1 ELSE 0 END) / count(res), 1) AS finish_rate_pct
""", {"constructorId": 3, "yearFrom": 2014, "yearTo": 2021})

add("constructor_stats", "season_drivers", ["constructor", "season"],
    "Who drove for McLaren in 2021 and how many points did each score?",
    f"""
MATCH (c:Constructor {{constructorId: $constructorId}})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:BY_DRIVER]->(d:Driver),
      (res)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}})
RETURN {DRIVER} AS driver, sum(res.points) AS points, count(res) AS races
ORDER BY points DESC
LIMIT 10
""", {"constructorId": 1, "year": 2021})

# ---------------------------------------------------------------- 3. head_to_head
# Teammate pattern: both Results in the same Race and for the same Constructor.
H2H_TEAMMATES = """
MATCH (da:Driver {driverId: $driverAId})<-[:BY_DRIVER]-(a:Result)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(b:Result)-[:BY_DRIVER]->(db:Driver {driverId: $driverBId}),
      (a)-[:FOR_CONSTRUCTOR]->(c:Constructor)<-[:FOR_CONSTRUCTOR]-(b)
WHERE ($constructorId IS NULL OR c.constructorId = $constructorId) AND """ + YEAR_RANGE

add("head_to_head", "finish_ahead", ["driver A", "driver B", "(optional) constructor", "(optional) years"],
    "Who finished ahead more often, Norris or Ricciardo at McLaren?",
    H2H_TEAMMATES + """
RETURN da.forename + ' ' + da.surname AS driver_a, db.forename + ' ' + db.surname AS driver_b,
       count(r) AS shared_races,
       sum(CASE WHEN a.positionOrder < b.positionOrder THEN 1 ELSE 0 END) AS a_finished_ahead,
       sum(CASE WHEN b.positionOrder < a.positionOrder THEN 1 ELSE 0 END) AS b_finished_ahead
""", {"driverAId": 846, "driverBId": 817, "constructorId": 1, "yearFrom": None, "yearTo": None})

add("head_to_head", "points_comparison", ["driver A", "driver B", "(optional) constructor", "(optional) years"],
    "Who scored more points as teammates, Hamilton or Rosberg?",
    H2H_TEAMMATES + """
RETURN da.forename + ' ' + da.surname AS driver_a, db.forename + ' ' + db.surname AS driver_b,
       count(r) AS shared_races, sum(a.points) AS points_a, sum(b.points) AS points_b
""", {"driverAId": 1, "driverBId": 3, "constructorId": None, "yearFrom": None, "yearTo": None})

add("head_to_head", "qualifying_battle", ["driver A", "driver B", "(optional) constructor", "(optional) years"],
    "Who out-qualified whom more often, Leclerc or Sainz?",
    """
MATCH (da:Driver {driverId: $driverAId})<-[:BY_DRIVER]-(a:Qualifying)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(b:Qualifying)-[:BY_DRIVER]->(db:Driver {driverId: $driverBId}),
      (a)-[:FOR_CONSTRUCTOR]->(c:Constructor)<-[:FOR_CONSTRUCTOR]-(b)
WHERE ($constructorId IS NULL OR c.constructorId = $constructorId) AND ($yearFrom IS NULL OR r.year >= $yearFrom) AND ($yearTo IS NULL OR r.year <= $yearTo)
RETURN da.forename + ' ' + da.surname AS driver_a, db.forename + ' ' + db.surname AS driver_b,
       count(r) AS shared_sessions,
       sum(CASE WHEN a.position < b.position THEN 1 ELSE 0 END) AS a_qualified_ahead,
       sum(CASE WHEN b.position < a.position THEN 1 ELSE 0 END) AS b_qualified_ahead
""", {"driverAId": 844, "driverBId": 832, "constructorId": 6, "yearFrom": None, "yearTo": None})

add("head_to_head", "at_circuit", ["driver A", "driver B", "circuit"],
    "Who finished ahead more often at Silverstone, Hamilton or Verstappen?",
    f"""
MATCH (da:Driver {{driverId: $driverAId}})<-[:BY_DRIVER]-(a:Result)-[:FOR_RACE]->(r:Race)<-[:FOR_RACE]-(b:Result)-[:BY_DRIVER]->(db:Driver {{driverId: $driverBId}}),
      (r)-[:AT_CIRCUIT]->(c:Circuit {{circuitId: $circuitId}})
RETURN da.forename + ' ' + da.surname AS driver_a, db.forename + ' ' + db.surname AS driver_b,
       c.circuit_name AS circuit, count(r) AS shared_races,
       sum(CASE WHEN a.positionOrder < b.positionOrder THEN 1 ELSE 0 END) AS a_finished_ahead,
       sum(CASE WHEN b.positionOrder < a.positionOrder THEN 1 ELSE 0 END) AS b_finished_ahead
""", {"driverAId": 1, "driverBId": 830, "circuitId": 9})

# ---------------------------------------------------------------- 4. race_result
add("race_result", "podium", ["race (Grand Prix name + year)"],
    "Who was on the podium at the 2019 German Grand Prix?",
    """
MATCH (res:Result)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}),
      (res)-[:BY_DRIVER]->(d:Driver), (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WHERE res.position <= 3
RETURN r.name AS race, r.year AS year, toString(r.date) AS date, res.position AS position,
       d.forename + ' ' + d.surname AS driver, c.constructor_name AS constructor
ORDER BY position
LIMIT 3
""", {"raceName": "German Grand Prix", "year": 2019})

add("race_result", "winner_by_circuit_year", ["circuit", "season"],
    "Who won at Monza in 2021?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(ci:Circuit {{circuitId: $circuitId}}),
      (r)-[:IN_SEASON]->(s:Season {{year: $year}}),
      (res)-[:BY_DRIVER]->(d:Driver), (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WHERE res.position = 1
RETURN r.name AS race, toString(r.date) AS date, ci.circuit_name AS circuit,
       {DRIVER} AS winner, c.constructor_name AS constructor
ORDER BY r.date
LIMIT 5
""", {"circuitId": 14, "year": 2021})

add("race_result", "fastest_lap", ["race (Grand Prix name + year)"],
    "Who set the fastest lap at the 2021 Italian Grand Prix?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race {{name: $raceName, year: $year}}), (res)-[:BY_DRIVER]->(d:Driver)
WHERE res.rank = 1
RETURN r.name AS race, r.year AS year, {DRIVER} AS driver, res.fastestLapTime AS fastest_lap_time,
       res.fastestLapSpeed AS fastest_lap_speed, res.fastestLap AS lap_number
LIMIT 1
""", {"raceName": "Italian Grand Prix", "year": 2021})

add("race_result", "retirements", ["race (Grand Prix name + year)"],
    "Who retired from the 2021 British Grand Prix and why?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race {{name: $raceName, year: $year}}),
      (res)-[:BY_DRIVER]->(d:Driver), (res)-[:FOR_CONSTRUCTOR]->(c:Constructor), (res)-[:HAS_STATUS]->(st:Status)
WHERE NOT {FINISHED}
RETURN {DRIVER} AS driver, c.constructor_name AS constructor, st.status AS reason, res.laps AS laps_completed
ORDER BY laps_completed DESC
LIMIT 25
""", {"raceName": "British Grand Prix", "year": 2021})

# ---------------------------------------------------------------- 5. circuit_info
add("circuit_info", "location", ["circuit"],
    "Where is Silverstone located?",
    """
MATCH (c:Circuit {circuitId: $circuitId})
RETURN c.circuit_name AS circuit, c.circuit_loc AS location, c.circuit_country AS country,
       c.lat AS latitude, c.lng AS longitude, c.alt AS altitude_m
LIMIT 1
""", {"circuitId": 9})

add("circuit_info", "races_held", ["circuit", "(optional) years"],
    "How many Grands Prix were held at Monza?",
    """
MATCH (r:Race)-[:AT_CIRCUIT]->(c:Circuit {circuitId: $circuitId})
WHERE ($yearFrom IS NULL OR r.year >= $yearFrom) AND ($yearTo IS NULL OR r.year <= $yearTo)
RETURN c.circuit_name AS circuit, count(r) AS races_held, min(r.year) AS first_year, max(r.year) AS last_year
""", {"circuitId": 14, "yearFrom": None, "yearTo": None})

add("circuit_info", "most_wins", ["circuit"],
    "Which driver has won the most races at Silverstone?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(c:Circuit {{circuitId: $circuitId}}),
      (res)-[:BY_DRIVER]->(d:Driver)
WHERE res.position = 1
RETURN c.circuit_name AS circuit, {DRIVER} AS driver, count(res) AS wins
ORDER BY wins DESC
LIMIT 5
""", {"circuitId": 9})

add("circuit_info", "circuits_in_country", ["country"],
    "Which circuits are in Italy and how many races were held at each?",
    """
MATCH (c:Circuit {circuit_country: $country})<-[:AT_CIRCUIT]-(r:Race)
RETURN c.circuit_name AS circuit, c.circuit_loc AS location, count(r) AS races_held
ORDER BY races_held DESC
LIMIT 15
""", {"country": "Italy"})

# ---------------------------------------------------------------- 6. qualifying
add("qualifying", "pole_count", ["driver", "(optional) years"],
    "How many pole positions did Hamilton take before 2020?",
    f"""
MATCH (q:Qualifying)-[:BY_DRIVER]->(d:Driver {{driverId: $driverId}}), (q)-[:FOR_RACE]->(r:Race)
WHERE q.position = 1 AND {YEAR_RANGE}
RETURN {DRIVER} AS driver, count(q) AS poles, min(r.year) AS first_pole_year, max(r.year) AS last_pole_year
""", {"driverId": 1, "yearFrom": None, "yearTo": 2019})

add("qualifying", "pole_sitter", ["race (Grand Prix name + year)"],
    "Who started on pole at the 2021 Italian Grand Prix?",
    """
MATCH (q:Qualifying)-[:FOR_RACE]->(r:Race {name: $raceName, year: $year}),
      (q)-[:BY_DRIVER]->(d:Driver), (q)-[:FOR_CONSTRUCTOR]->(c:Constructor)
WHERE q.position = 1
RETURN r.name AS race, r.year AS year, d.forename + ' ' + d.surname AS driver,
       c.constructor_name AS constructor, q.q1 AS q1, q.q2 AS q2, q.q3 AS q3
LIMIT 1
""", {"raceName": "Italian Grand Prix", "year": 2021})

add("qualifying", "q3_appearances", ["driver", "season"],
    "In how many qualifying sessions did Norris reach Q3 in 2021?",
    f"""
MATCH (q:Qualifying)-[:BY_DRIVER]->(d:Driver {{driverId: $driverId}}),
      (q)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}})
RETURN {DRIVER} AS driver, s.year AS season,
       sum(CASE WHEN q.q3 IS NOT NULL THEN 1 ELSE 0 END) AS q3_appearances,
       count(q) AS qualifying_sessions
""", {"driverId": 846, "year": 2021})

add("qualifying", "grid_gains", ["circuit", "season"],
    "Which driver gained the most places from the grid at Monza in 2021?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:AT_CIRCUIT]->(ci:Circuit {{circuitId: $circuitId}}),
      (r)-[:IN_SEASON]->(s:Season {{year: $year}}), (res)-[:BY_DRIVER]->(d:Driver)
WHERE res.grid > 0 AND res.position IS NOT NULL
RETURN r.name AS race, {DRIVER} AS driver, res.grid AS grid, res.position AS finish,
       res.grid - res.position AS places_gained
ORDER BY places_gained DESC
LIMIT 5
""", {"circuitId": 14, "year": 2021})

# ---------------------------------------------------------------- 7. pit_stops
add("pit_stops", "fastest_in_race", ["race (Grand Prix name + year)"],
    "What was the fastest pit stop at the 2023 Qatar Grand Prix?",
    f"""
MATCH (p:PitStop)-[:FOR_RACE]->(r:Race {{name: $raceName, year: $year}}), (p)-[:BY_DRIVER]->(d:Driver)
WHERE p.milliseconds IS NOT NULL
RETURN r.name AS race, {DRIVER} AS driver, p.stop AS stop_number, p.lap AS lap,
       p.duration AS duration, p.milliseconds AS milliseconds
ORDER BY p.milliseconds ASC
LIMIT 3
""", {"raceName": "Qatar Grand Prix", "year": 2023})

add("pit_stops", "stops_per_driver", ["race (Grand Prix name + year)"],
    "How many pit stops did each driver make at the 2021 Hungarian Grand Prix?",
    f"""
MATCH (p:PitStop)-[:FOR_RACE]->(r:Race {{name: $raceName, year: $year}}), (p)-[:BY_DRIVER]->(d:Driver)
RETURN r.name AS race, {DRIVER} AS driver, count(p) AS stops
ORDER BY stops DESC, driver
LIMIT 25
""", {"raceName": "Hungarian Grand Prix", "year": 2021})

add("pit_stops", "driver_avg_season", ["driver", "season"],
    "What was Verstappen's average pit stop time in 2022?",
    f"""
MATCH (p:PitStop)-[:BY_DRIVER]->(d:Driver {{driverId: $driverId}}),
      (p)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}})
WHERE p.milliseconds IS NOT NULL AND p.milliseconds < 60000
RETURN {DRIVER} AS driver, s.year AS season, count(p) AS stops,
       round(avg(p.milliseconds) / 1000.0, 3) AS avg_seconds,
       round(percentileCont(p.milliseconds, 0.5) / 1000.0, 3) AS median_seconds,
       round(min(p.milliseconds) / 1000.0, 3) AS fastest_seconds
""", {"driverId": 830, "year": 2022})

add("pit_stops", "fastest_in_season", ["season"],
    "What were the fastest pit stops of 2023?",
    f"""
MATCH (p:PitStop)-[:BY_DRIVER]->(d:Driver), (p)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}})
WHERE p.milliseconds IS NOT NULL
RETURN r.name AS race, {DRIVER} AS driver, p.lap AS lap, p.duration AS duration, p.milliseconds AS milliseconds
ORDER BY p.milliseconds ASC
LIMIT 5
""", {"year": 2023})

# ---------------------------------------------------------------- 8. season_history
add("season_history", "points_leader", ["season"],
    "Who scored the most points in 2021?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}}), (res)-[:BY_DRIVER]->(d:Driver)
RETURN s.year AS season, {DRIVER} AS driver, sum(res.points) AS points,
       sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
ORDER BY points DESC
LIMIT 3
""", {"year": 2021})

add("season_history", "drivers_in_season", ["season"],
    "How many drivers competed in 2020?",
    f"""
MATCH (res:Result)-[:FOR_RACE]->(r:Race)-[:IN_SEASON]->(s:Season {{year: $year}}), (res)-[:BY_DRIVER]->(d:Driver)
RETURN s.year AS season, count(DISTINCT d) AS drivers,
       collect(DISTINCT {DRIVER})[..40] AS driver_names
""", {"year": 2020})

add("season_history", "team_changes", ["year A", "year B"],
    "Which drivers changed teams between 2020 and 2021?",
    f"""
MATCH (d:Driver)<-[:BY_DRIVER]-(r1:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(:Season {{year: $yearA}}),
      (r1)-[:FOR_CONSTRUCTOR]->(c1:Constructor)
WITH d, collect(DISTINCT c1.constructor_name) AS teams_a
MATCH (d)<-[:BY_DRIVER]-(r2:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(:Season {{year: $yearB}}),
      (r2)-[:FOR_CONSTRUCTOR]->(c2:Constructor)
WITH d, teams_a, collect(DISTINCT c2.constructor_name) AS teams_b
WHERE none(t IN teams_b WHERE t IN teams_a)
RETURN {DRIVER} AS driver, teams_a AS teams_in_year_a, teams_b AS teams_in_year_b
ORDER BY driver
LIMIT 30
""", {"yearA": 2020, "yearB": 2021})

add("season_history", "calendar", ["season"],
    "Which Grands Prix were held in 2022?",
    """
MATCH (r:Race)-[:IN_SEASON]->(s:Season {year: $year}), (r)-[:AT_CIRCUIT]->(c:Circuit)
RETURN s.year AS season, r.round AS round, r.name AS race, toString(r.date) AS date,
       c.circuit_name AS circuit, c.circuit_country AS country
ORDER BY r.round
LIMIT 30
""", {"year": 2022})

TEMPLATES = T


def intent_of(key):
    return TEMPLATES[key]["intent"]


def with_defaults(params):
    return {**OPTIONAL_DEFAULTS, **params}


def run_template(session, key, params):
    """Execute a template with parameters only; returns a list of dict rows."""
    t = TEMPLATES[key]
    result = session.run(t["cypher"], with_defaults(params))
    return [dict(r) for r in result]


def browser_block(key):
    t = TEMPLATES[key]
    lines = [f":param {k} => {('null' if v is None else repr(v).replace(chr(39), chr(34)))};"
             for k, v in with_defaults(t["example"]).items()]
    return "\n".join(lines) + "\n\n" + t["cypher"] + ";"


def main(argv):
    if argv and argv[0] == "--browser":
        for key in argv[1:] or TEMPLATES:
            print(f"// ---- {key}\n{browser_block(key)}\n")
        return
    from neo4j import GraphDatabase
    from Create_kg import read_config
    cfg = read_config()
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    keys = argv or list(TEMPLATES)
    try:
        with driver.session(database=cfg.get("DATABASE") or None) as session:
            for key in keys:
                rows = run_template(session, key, TEMPLATES[key]["example"])
                print(f"\n== {key}  ({len(rows)} rows)  e.g. {TEMPLATES[key]['question']}")
                for row in rows[:5]:
                    print("  ", row)
    finally:
        driver.close()


if __name__ == "__main__":
    main(sys.argv[1:])
